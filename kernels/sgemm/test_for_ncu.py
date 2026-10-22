import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
# 将 CUDA 源码即时编译并加载为 Python 扩展模块。
lib = load(
    name="gemm_lib",
    sources=["sgemm.cu", "cublas.cu", "sgemm_tf32.cu"],
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

# 此脚本用于 Nsight Compute 采集：默认零次预热、一次执行，且没有测量 PyTorch 基准。
# 下面的 baseline 是占位数，此时打印的 speedup 不能作为有效加速比。
baseline = 1e-6


def benchmark(op, a, b, c=None, warmup=0, rep=1, prefix="torch"):
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

    if prefix == "torch":
        global baseline
        baseline = duration
        print(f"{prefix:30s} mean time: {duration / rep * 1000:.6f} ms")
    else:
        speedup = baseline / duration
        print(
            f"{prefix:30s} mean time: {duration / rep * 1000:.6f} ms, speedup: {speedup:.2f}"
        )


def run_fp32():
    n, m, k = 4096, 4096, 4096
    print("#" * 100)
    print(f"m: {m}, n: {n}, k: {k}")
    # 矩阵乘法：a × b = c。
    a = torch.randn(m, k).float().cuda()
    b = torch.randn(k, n).float().cuda()
    c = torch.zeros(m, n).float().cuda()

    # CUDA 核心路径。
    # benchmark(partial(torch.matmul, out=c), a, b)
    c_cublas = torch.zeros_like(c)
    benchmark(lib.sgemm_cublas, a, b, c_cublas, prefix="sgemm_cublas")
    c_my = torch.zeros_like(c)
    # benchmark(lib.sgemm_naive, a, b, c_my, prefix="sgemm_naive")
    benchmark(lib.sgemm_tiling, a, b, c_my, prefix="sgemm_tiling")
    benchmark(lib.sgemm_at_tiling, a, b, c_my, prefix="sgemm_at_tiling")
    benchmark(lib.sgemm_at_bcf_swizzling, a, b, c_my, prefix="sgemm_at_bcf_swizzling")
    benchmark(
        lib.sgemm_at_bcf_swizzling_rw, a, b, c_my, prefix="sgemm_at_bcf_swizzling_rw"
    )
    benchmark(
        lib.sgemm_at_bcf_swizzling_dbf_rw,
        a,
        b,
        c_my,
        prefix="sgemm_at_bcf_swizzling_dbf_rw",
    )


def run_tf32():
    n, m, k = 4096, 4096, 4096
    print("#" * 100)
    print(f"m: {m}, n: {n}, k: {k}")
    # 矩阵乘法：a × b = c。
    a = torch.randn(m, k).float().cuda()
    b = torch.randn(k, n).float().cuda()
    c = torch.zeros(m, n).float().cuda()

    # CUDA 核心路径。
    # benchmark(partial(torch.matmul, out=c), a, b)
    c_cublas = torch.zeros_like(c)
    benchmark(lib.sgemm_cublas_tf32, a, b, c_cublas, prefix="sgemm_cublas_tf32")
    c_my = torch.zeros_like(c)
    # benchmark(lib.sgemm_naive, a, b, c_my, prefix="sgemm_naive")
    benchmark(lib.sgemm_tf32_bt, a, b, c_my, prefix="sgemm_tf32_bt")
    benchmark(
        lib.sgemm_tf32_bt_swizzle,
        a,
        b,
        c_my,
        prefix="sgemm_tf32_bt_swizzle",
    )
    benchmark(
        lib.sgemm_tf32_bt_swizzle_dbf,
        a,
        b,
        c_my,
        prefix="sgemm_tf32_bt_swizzle_dbf",
    )
    benchmark(
        lib.sgemm_tf32_swizzle_bcf,
        a,
        b,
        c_my,
        prefix="sgemm_tf32_swizzle_bcf",
    )
    benchmark(
        lib.sgemm_tf32_swizzle_bcf_dbf,
        a,
        b,
        c_my,
        prefix="sgemm_tf32_swizzle_bcf_dbf",
    )


if __name__ == "__main__":
    # run_fp32()
    run_tf32()
