# 中文阅读提示：此脚本会即时编译 CUDA 扩展；下面是原有测试配置，本次未执行 GPU 测试。
# 本入口供 Nsight Compute 采集使用；分析器运行会改变计时条件，不能直接与普通计时混用。
# 各入口的缓存容量和 dtype 限制见 softmax.cu；有边界判断不意味着任意行宽都支持。
import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
# 将 CUDA 源码即时编译并加载为 Python 扩展模块。
lib = load(
    name="softmax_lib",
    sources=["softmax.cu"],
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


def benchmark(op, a, b=None, warmup=0, rep=1, prefix="torch"):
    if b is not None:
        # 预热，减少首次运行对正式计时的影响。
        for i in range(warmup):
            res = op(a, b)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(rep):
            res = op(a, b)
    else:
        # 预热，减少首次运行对正式计时的影响。
        for i in range(warmup):
            res = op(a)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(rep):
            res = op(a)
    torch.cuda.synchronize()
    if prefix == "torch":
        global baseline
        duration = time.perf_counter() - start
        baseline = duration
        print(f"{prefix:30s} mean time: {duration / rep * 1000:.6f} ms")
    else:
        duration = time.perf_counter() - start
        speedup = baseline / duration
        print(
            f"{prefix:30s} mean time: {duration / rep * 1000:.6f} ms, speedup: {speedup:.2f}"
        )
    return res


def diff_check(a, b, prefix="torch", eps=1e-2):
    if not torch.allclose(a, b, atol=eps, rtol=eps):
        print(f"{prefix} result diff: {torch.mean(torch.abs(a - b)).item()}")
    assert torch.allclose(a, b, atol=eps, rtol=eps), "result diff"


def test_small():
    bs = [2048]
    sz = [8192]
    torch.manual_seed(42)
    for n in bs:
        for m in sz:
            print("#" * 100)
            print(f"bs: {n}, hidden_size: {m}")
            a = torch.randn(n, m).float().cuda()
            b = torch.empty(n, m).float().cuda()

            benchmark(partial(torch.softmax, dim=1, out=b), a)
            b_my = torch.empty_like(a)
            benchmark(lib.softmax, a, b_my, prefix="softmax")
            # print(b, b_my)
            diff_check(b, b_my, prefix="softmax")

            benchmark(lib.softmax_fp32x4, a, b_my, prefix="softmax_fp32x4")
            # print(b, b_my)
            diff_check(b, b_my, prefix="softmax_fp32x4")

            a = a.half()
            b = b.half()
            b_my = b_my.half()
            benchmark(partial(torch.softmax, dim=1, out=b), a)
            benchmark(lib.softmax, a, b_my, prefix="softmax_fp16")
            diff_check(b, b_my, prefix="softmax_fp16")

            benchmark(
                lib.softmax_fp16x8_packed, a, b_my, prefix="softmax_fp16x8_packed"
            )
            diff_check(b, b_my, prefix="softmax_fp16x8_packed")

            benchmark(lib.softmax_arbitrary, a, b_my, prefix="softmax_arbitrary")

            diff_check(b, b_my, prefix="softmax_arbitrary")

def test_large():
    ns = [26]
    ms = [32768]
    for n in ns:
        for m in ms:
            print("#" * 100)
            print(f"bs: {n}, hidden_size: {m}")
            a = torch.randn(n, m).half().cuda()
            b = torch.zeros(n, m).half().cuda()

            benchmark(partial(torch.softmax, dim=1, out=b), a)
            b_my = torch.zeros_like(a)

            if m <= 32768:
                benchmark(lib.softmax_medium, a, b_my, prefix="softmax_medium")
                diff_check(b, b_my, prefix="softmax_medium")
            if m <= 114688:
                benchmark(lib.softmax_extreme, a, b_my, prefix="softmax_extreme")
                diff_check(b, b_my, prefix="softmax_extreme")

            benchmark(lib.softmax_arbitrary, a, b_my, prefix="softmax_arbitrary")
            # print(b, b_my)
            diff_check(b, b_my, prefix="softmax_arbitrary")

            benchmark(lib.softmax_splitk, a, b_my, prefix="softmax_splitk")
            diff_check(b, b_my, prefix="softmax_splitk")

if __name__ == "__main__":
    # 运行下面列出的核函数测试配置。
    # test_small()
    test_large()
