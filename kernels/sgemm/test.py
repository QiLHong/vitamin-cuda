import time
from functools import partial
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
current_dir = Path(__file__).parent.resolve()
# 将 CUDA 源码即时编译并加载为 Python 扩展模块。
lib = load(
    name="gemm_lib",
    sources=[str(current_dir / "sgemm.cu"), str(current_dir / "cublas.cu"), str(current_dir / "sgemm_tf32.cu")],
    extra_cuda_cflags=common_flags
    + [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-Xptxas -v"
    ],
    extra_cflags=common_flags,
    verbose=True,
)

baseline = None


# 用主机时钟测重复调用的总时间，前后同步后除以 rep；包含这一调用方式下的调度开销。
# 正确性由后续 diff_check 单独比较；完成计时并不自动证明结果正确。
def benchmark(op, a, b, c=None, warmup=10, rep=200, prefix="torch"):
    if c is not None:
        # 预热，减小首次执行开销对计时的影响。
        for i in range(warmup):
            op(a, b, c)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(rep):
            op(a, b, c)
        torch.cuda.synchronize()
    else:
        # 预热，减小首次执行开销对计时的影响。
        for i in range(warmup):
            op(a, b)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(rep):
            op(a, b)
        torch.cuda.synchronize()

    duration = time.perf_counter() - start

    tflops = a.shape[0]*b.shape[0]*b.shape[1]*2*rep / 1e12 / duration
    if prefix == "torch":
        global baseline
        baseline = duration
        print(f"{prefix:40s} mean time: {duration / rep * 1000:8.6f} ms, {tflops:.2f} tflops")
    else:
        speedup = baseline / duration
        print(
            f"{prefix:40s} mean time: {duration / rep * 1000:8.6f} ms, speedup: {speedup:.2f}, tflops: {tflops:.2f}"
        )


def diff_check(a, b, prefix="torch", eps=1e-3):
    if "tf32" in prefix:
        eps = 0.1
    if not torch.allclose(a, b, atol=eps, rtol=eps):
        print(f"{prefix} result diff: {torch.mean(torch.abs(a - b)).item()}")
    assert torch.allclose(a, b, atol=eps, rtol=eps), "result diff"


def test_all():
    # 测试各核函数。
    ns = [256, 512, 1024, 4096, 8192]
    torch.manual_seed(42)
    for i, m in enumerate(ns):
        for k in ns[i-1:i+1]:
            for n in ns[i-1:i+1]:
                print("#" * 100)
                print(f"m: {m}, n: {n}, k: {k}")
                # 矩阵乘法：a × b = c。
                a = torch.randn(m, k).float().cuda()
                b = torch.randn(k, n).float().cuda()
                c = torch.zeros(m, n).float().cuda()
                print("#" * 45 + "CUDA CORE" + "#" * 46)
                # CUDA 核心路径。
                benchmark(partial(torch.matmul, out=c), a, b)
                c_cublas = torch.zeros_like(c)
                benchmark(lib.sgemm_cublas, a, b, c_cublas, prefix="sgemm_cublas")
                diff_check(c, c_cublas, prefix="sgemm_cublas")
                c_my = torch.zeros_like(c)
                # benchmark(lib.sgemm_naive, a, b, c_my, prefix="sgemm_naive")
                # diff_check(c, c_my, prefix="sgemm_naive")
                benchmark(lib.sgemm_tiling, a, b, c_my, prefix="sgemm_tiling")
                diff_check(c, c_my, prefix="sgemm_tiling")
                benchmark(lib.sgemm_at_tiling, a, b, c_my, prefix="sgemm_at_tiling")
                diff_check(c, c_my, prefix="sgemm_at_tiling")
                benchmark(lib.sgemm_at_bcf_swizzling, a, b, c_my, prefix="sgemm_at_bcf_swizzling")
                diff_check(c, c_my, prefix="sgemm_at_bcf_swizzling")
                benchmark(lib.sgemm_at_bcf_swizzling_rw, a, b, c_my, prefix="sgemm_at_bcf_swizzling_rw")
                diff_check(c, c_my, prefix="sgemm_at_bcf_swizzling_rw")

                print("#" * 44 + "Tensor Core" + "#" * 45)
                # Tensor Core 路径。
                benchmark(
                    lib.sgemm_cublas_tf32,
                    a,
                    b,
                    c_cublas,
                    prefix="sgemm_cublas_tf32",
                )
                diff_check(c, c_cublas, prefix="sgemm_cublas_tf32")

def test_4096():
    m, n, k = [4096] * 3
    print("#" * 100)
    print(f"n: {n}, m: {m}, k: {k}")
    # 矩阵乘法：a × b = c。
    a = torch.randn(m, k).float().cuda()
    b = torch.randn(k, n).float().cuda()
    c = torch.zeros(m, n).float().cuda()

    # CUDA 核心路径。
    benchmark(partial(torch.matmul, out=c), a, b)
    c_cublas = torch.zeros_like(c)
    benchmark(lib.sgemm_cublas, a, b, c_cublas, prefix="sgemm_cublas")
    diff_check(c, c_cublas, prefix="sgemm_cublas")
    c_my = torch.zeros_like(c)
    # benchmark(lib.sgemm_naive, a, b, c_my, prefix="sgemm_naive")
    # diff_check(c, c_my, prefix="sgemm_naive")
    benchmark(lib.sgemm_tiling, a, b, c_my, prefix="sgemm_tiling")
    diff_check(c, c_my, prefix="sgemm_tiling")
    benchmark(lib.sgemm_at_tiling, a, b, c_my, prefix="sgemm_at_tiling")
    diff_check(c, c_my, prefix="sgemm_at_tiling")
    benchmark(lib.sgemm_at_bcf_swizzling, a, b, c_my, prefix="sgemm_at_bcf_swizzling")
    diff_check(c, c_my, prefix="sgemm_at_bcf_swizzling")
    benchmark(lib.sgemm_at_bcf_swizzling_rw, a, b, c_my, prefix="sgemm_at_bcf_swizzling_rw")
    diff_check(c, c_my, prefix="sgemm_at_bcf_swizzling_rw")
    benchmark(lib.sgemm_at_bcf_swizzling_dbf_rw, a, b, c_my, prefix="sgemm_at_bcf_swizzling_dbf_rw")
    diff_check(c, c_my, prefix="sgemm_at_bcf_swizzling_dbf_rw")

    # Tensor Core 路径。
    benchmark(
        lib.sgemm_cublas_tf32,
        a,
        b,
        c_cublas,
        prefix="sgemm_cublas_tf32",
    )
    diff_check(c, c_cublas, prefix="sgemm_cublas_tf32")

def test_tf32_4096():
    m, n, k = [4096] * 3
    print("#" * 100)
    print(f"n: {n}, m: {m}, k: {k}")
    # 矩阵乘法：a × b = c。
    a = torch.randn(m, k).float().cuda()
    b = torch.randn(k, n).float().cuda()
    c = torch.zeros(m, n).float().cuda()
    c_cublas = torch.zeros_like(c)
    # CUDA 核心路径。
    benchmark(partial(torch.matmul, out=c), a, b)
    c_cublas = torch.zeros_like(c)

    # Tensor Core 路径。
    benchmark(
        lib.sgemm_cublas_tf32,
        a,
        b,
        c_cublas,
        prefix="sgemm_cublas_tf32",
    )
    diff_check(c, c_cublas, prefix="sgemm_cublas_tf32")
    c_my = torch.zeros_like(c)
    benchmark(
        lib.sgemm_tf32_bt,
        a,
        b,
        c_my,
        prefix="sgemm_tf32_bt",
    )
    diff_check(c, c_my, prefix="sgemm_tf32_bt")
    benchmark(
        lib.sgemm_tf32_bt_swizzle,
        a,
        b,
        c_my,
        prefix="sgemm_tf32_bt_swizzle",
    )
    diff_check(c, c_my, prefix="sgemm_tf32_bt_swizzle")
    benchmark(
        lib.sgemm_tf32_bt_swizzle_dbf,
        a,
        b,
        c_my,
        prefix="sgemm_tf32_bt_swizzle_dbf",
    )
    diff_check(c, c_my, prefix="sgemm_tf32_bt_swizzle_dbf")
    benchmark(
        lib.sgemm_tf32_swizzle_bcf,
        a,
        b,
        c_my,
        prefix="sgemm_tf32_swizzle_bcf",
    )
    diff_check(c, c_my, prefix="sgemm_tf32_swizzle_bcf")
    benchmark(
        lib.sgemm_tf32_swizzle_bcf_dbf,
        a,
        b,
        c_my,
        prefix="sgemm_tf32_swizzle_bcf_dbf",
    )
    diff_check(c, c_my, prefix="sgemm_tf32_swizzle_bcf_dbf")

if __name__ == "__main__":
    # test_all()
    # test_4096()
    test_tf32_4096()
