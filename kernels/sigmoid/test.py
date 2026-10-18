# 中文导读：本脚本比较 PyTorch 与手写 Sigmoid，先即时编译，再测量，最后校验结果。
# 本次只改注释；原测试的输出、形状、阈值和编译选项保持不变。
# 请先读同目录 readme.md，特别是每种向量化实现的尺寸限制，不要直接换成任意奇数。
import numpy as np
from functools import partial
from pathlib import Path

import torch
from flashinfer.testing.utils import bench_gpu_time
from torch.utils.cpp_extension import load

# 关闭自动求导记录，只测试前向；这不等于为扩展注册了梯度公式。
torch.set_grad_enabled(False)

# 使用 C++17 与优化；源码包装使用 if constexpr 等编译期语法。
common_flags = ["-O3", "-std=c++17"]
current_dir = Path(__file__).parent.resolve()
# load() 将本目录 CUDA 源码编译成动态扩展并加载，不是直接解释执行 .cu 文件。
# PyTorch 外还需匹配的 nvcc/C++ 编译器与 Ninja；原脚本另外导入 FlashInfer 计时工具。
# -U 取消对应禁用宏；--use_fast_math 可能影响数值结果，首次编译时间不属于核函数耗时。
lib = load(
    name="sigmoid_lib",
    sources=[str(current_dir / "sigmoid.cu")],
    extra_cuda_cflags=common_flags
    + [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=common_flags,
    verbose=True,
)


# 每一轮先测同 dtype 的 PyTorch，才有用于计算加速比的基线。
baseline = None


# 自定义函数接收输入 a 和预分配输出 b；partial 则让 PyTorch 预先绑定 out 参数。
# warmup 是预热次数，rep 是重复次数；默认设置不等于所有 GPU 上的最佳测试方案。
def benchmark(op, a, b=None, warmup=10, rep=100, prefix="torch"):
    global baseline
    input_args = (a, b) if b is not None else (a,)

    # 这里测设备时间，并使用冷 L2 缓存；不要与不同缓存/Graph 配置的结果直接比较。
    times = bench_gpu_time(
        fn=op,
        input_args=input_args,
        dry_run_iters=warmup,
        repeat_iters=rep,
        enable_cupti=False,
        use_cuda_graph=False,
        cold_l2_cache=True,
    )
    # 实际统计中位数，尽管原英文输出写着 mean time；保留字符串以避免行为改动。
    avg_duration = float(np.median(times))

    if prefix == "torch":
        baseline = avg_duration
        print(f"{prefix:30s} mean time: {avg_duration:8.6f} ms")
    else:
        # 大于 1 表示这一轮快于基线，小于 1 表示更慢；不是精度指标。
        speedup = baseline / avg_duration
        print(f"{prefix:30s} mean time: {avg_duration:8.6f} ms, speedup: {speedup:.2f}")


# 最大绝对误差检查；取回 .item() 会等待相关 GPU 结果。Python -O 会禁用 assert。
# 本脚本 FP32 用默认 1e-5，FP16 显式用 1e-3；不要无依据放宽阈值掩盖实现问题。
def diff_check(a, b, prefix="torch", eps=1e-5):
    message = f"{prefix} result diff"
    assert torch.max(torch.abs(a - b)).item() < eps, message


if __name__ == "__main__":
    # 执行原有九组形状测试：这些整齐尺寸满足此处各向量化版本的整块约定。
    device = torch.device("cuda")
    sz = [1024, 2048, 4096]
    for n in sz:
        for m in sz:
            print("#" * 100)
            print(f"n: {n}, m: {m}")
            a = torch.randn(n, m).float().cuda()
            # empty_like 只分配输出，值尚未初始化；下面的 PyTorch 基准调用会写入 b。
            b = torch.empty_like(a).float().cuda()

            benchmark(partial(torch.sigmoid, out=b), a)
            b_my = torch.empty_like(a)

            benchmark(lib.sigmoid, a, b_my, prefix="sigmoid")
            # 原调试语句保持禁用，避免把打印和数据传回 CPU 的开销混入测量。
            # print(b, b_my)
            diff_check(b, b_my, prefix="sigmoid")
            benchmark(lib.sigmoid_fp32x4, a, b_my, prefix="sigmoid_fp32x4")
            diff_check(b, b_my, prefix="sigmoid_fp32x4")

            # 半精度阶段：真正转换张量 dtype，重新测量对应精度的 PyTorch 基线。
            # 不是仅重解释指针；中间运算精度差异要结合 CUDA 辅助函数理解。
            a = a.half()
            b = b.half()
            benchmark(partial(torch.sigmoid, out=b), a)
            b_my = b_my.half()

            benchmark(lib.sigmoid, a, b_my, prefix="sigmoid_half")
            diff_check(b, b_my, prefix="sigmoid_half", eps=1e-3)
            benchmark(lib.sigmoid_fp16x2, a, b_my, prefix="sigmoid_fp16x2")
            diff_check(b, b_my, prefix="sigmoid_fp16x2", eps=1e-3)
            benchmark(lib.sigmoid_fp16x8, a, b_my, prefix="sigmoid_fp16x8")
            diff_check(b, b_my, prefix="sigmoid_fp16x8", eps=1e-3)
            benchmark(
                lib.sigmoid_fp16x8_packed,
                a,
                b_my,
                prefix="sigmoid_fp16x8_packed",
            )
            diff_check(b, b_my, prefix="sigmoid_fp16x8_packed", eps=1e-3)
