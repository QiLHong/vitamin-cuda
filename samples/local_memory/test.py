import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
# 从当前目录即时编译 CUDA 扩展；-Xptxas -v 请求输出资源使用信息。
lib = load(
    name="load_lib",
    sources=["test.cu"],
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


if __name__ == "__main__":
    # 原始单块实验，依次启动三个入口但没有结果断言；主要用于查看编译输出。
    a = torch.randn(512).half().cuda()
    b = torch.zeros_like(a)
    lib.load_fp16x8_native(a, b)
    lib.load_fp16x8_bad(a, b)
    lib.load_fp16x8_good(a, b)
