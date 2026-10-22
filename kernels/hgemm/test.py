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
    name="hgemm_lib",
    sources=[str(current_dir / "hgemm.cu"), str(current_dir / "cublas.cu")],
    extra_cuda_cflags=common_flags
    + [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-Xptxas -v",
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

    tflops = a.shape[0] * b.shape[0] * b.shape[1] * 2 * rep / 1e12 / duration
    if prefix == "torch":
        global baseline
        baseline = duration
        print(
            f"{prefix:40s} mean time: {duration / rep * 1000:8.6f} ms, {tflops:.2f} tflops"
        )
    else:
        speedup = baseline / duration
        print(
            f"{prefix:40s} mean time: {duration / rep * 1000:8.6f} ms, speedup: {speedup:.2f}, tflops: {tflops:.2f}"
        )


def diff_check(a, b, prefix="torch", eps=0.016):
    if not torch.allclose(a, b, atol=eps, rtol=eps):
        diff = torch.abs(a - b)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        print(f"{prefix} result mean diff: {mean_diff:.6f}, max diff: {max_diff:.6f}")
    assert torch.allclose(a, b, atol=eps, rtol=eps), "result diff"


def test_all():
    # 测试各核函数。
    ns = [256, 512, 1024, 4096, 8192]
    torch.manual_seed(42)
    # for type in [torch.bfloat16, torch.float16]:
    for type in [torch.bfloat16]:
        for n in ns:
            print("#" * 100)
            print(f"m: {n}, n: {n}, k: {n}")
            # 矩阵乘法：a × b = c。
            a = torch.randn(n, n).to(type).cuda()
            b = torch.randn(n, n).to(type).cuda()
            c = torch.zeros(n, n).to(type).cuda()
            print("#" * 100)
            # PyTorch 与 cuBLAS 参考路径，具体硬件执行路径由库选择。
            benchmark(partial(torch.matmul, out=c), a, b)
            c_cublas = torch.zeros_like(c)
            benchmark(
                lib.hgemm_cublas,
                a,
                b,
                c_cublas,
                prefix="hgemm_cublas",
            )
            diff_check(c, c_cublas, prefix="hgemm_cublas")
            # c_my = torch.zeros_like(c)
            # benchmark(lib.hgemm_naive, a, b, c_my, prefix="hgemm_naive")
            # diff_check(c, c_my, prefix="hgemm_naive")


def test_4096():

    m, n, k = [4096] * 3
    print("#" * 100)
    print(f"n: {n}, m: {m}, k: {k}")
    # for type in [torch.bfloat16, torch.float16]:
    for type in [torch.bfloat16]:
        # 矩阵乘法：a × b = c。
        a = torch.randn(m, k).to(type).cuda()
        b = torch.randn(k, n).to(type).cuda()
        c = torch.zeros(m, n).to(type).cuda()

        benchmark(partial(torch.matmul, out=c), a, b)

        c_cublas = torch.zeros_like(c)
        benchmark(
            lib.hgemm_cublas,
            a,
            b,
            c_cublas,
            prefix="hgemm_cublas",
        )
        diff_check(c, c_cublas, prefix="hgemm_cublas")

        c_my = torch.zeros_like(c)

        benchmark(lib.hgemm_naive, a, b, c_my, prefix="hgemm_naive")
        diff_check(c, c_my, prefix="hgemm_naive")

        benchmark(lib.hgemm_bcf, a, b, c_my, prefix="hgemm_bcf")
        diff_check(c, c_my, prefix="hgemm_bcf")
        benchmark(lib.hgemm_bcf_dbf, a, b, c_my, prefix="hgemm_bcf_dbf")
        diff_check(c, c_my, prefix="hgemm_bcf_dbf")
        benchmark(lib.hgemm_bcf_dbf_rw, a, b, c_my, prefix="hgemm_bcf_dbf_rw")
        diff_check(c, c_my, prefix="hgemm_bcf_dbf_rw")


if __name__ == "__main__":
    # test_all()
    test_4096()
