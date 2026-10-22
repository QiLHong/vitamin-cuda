import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
# 将 CUDA 源码即时编译并加载为 Python 扩展模块。
lib = load(
    name="hgemm_sm120_lib",
    sources=["hgemm_sm120.cu", "cublas.cu"],
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
    extra_ldflags=["-L/usr/local/cuda/lib64/stubs", "-lcuda"],
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


def run():
    m, n, k = [4096] * 3
    print("#" * 100)
    print(f"n: {n}, m: {m}, k: {k}")
    # 矩阵乘法：a × b = c。
    a = torch.randn(m, k).bfloat16().cuda()
    b = torch.randn(k, n).bfloat16().cuda()
    c = torch.zeros(m, n).bfloat16().cuda()


    c_cublas = torch.zeros_like(c)
    benchmark(
        lib.hgemm_cublas,
        a,
        b,
        c_cublas,
        prefix="hgemm_cublas",
    )

    c_my = torch.zeros_like(c)
    benchmark(lib.hgemm_bcf_dbf_rw, a, b, c_my, prefix="hgemm_bcf_dbf_rw")
    benchmark(lib.hgemm_k_stages, a, b, c_my, prefix="hgemm_3_stages")
    benchmark(lib.hgemm_tma_r_k_stages_64, a, b, c_my, prefix="hgemm_tma_r_k_stages_64")
    benchmark(lib.hgemm_tma_r_k_stages_32, a, b, c_my, prefix="hgemm_tma_r_k_stages_32")


if __name__ == "__main__":
    run()
