# 中文导读：本脚本分为“即时编译扩展、测量耗时、与 PyTorch 比较结果”三部分。
# 运行前阅读同目录 readme.md。首次执行会编译 CUDA，不要把编译时间算成核函数耗时。
# 原测试使用 FlashInfer 的基准工具；只装 PyTorch 并不一定能运行本脚本。
# 本分支仅添加/翻译注释，保留原测试尺寸、阈值、参数和英文输出。
import time
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from flashinfer.testing.utils import bench_gpu_time, get_l2_cache_size
from torch.utils.cpp_extension import load

# 本例只测前向计算，关闭自动求导记录；这不等于为自定义算子实现了反向传播。
torch.set_grad_enabled(False)

# -O3 开启优化，-std=c++17 使用 C++17（源码中的 if constexpr 等需要它）。
common_flags = ["-O3", "-std=c++17"]
# 通过脚本自身的位置定位 .cu 文件，避免依赖终端的当前工作目录。
current_dir = Path(__file__).parent.resolve()
# 将 CUDA 源码即时编译并加载为 Python 模块；load() 不是“直接解释执行 CUDA”。
# name 决定扩展模块名，sources 指定编译输入；verbose=True 会显示编译过程。
# -U... 取消相应禁用宏；--use_fast_math 可能改变部分浮点运算精度，不能当作无代价开关。
lib = load(
    name="elementwise_lib",
    sources=[str(current_dir / "elementwise.cu")],
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


# op 是可调用对象。原生 PyTorch 可返回结果，自定义入口则将结果写进预分配的 c。
# warmup 用于预热，rep 控制重复测量；这里通过工具采集 GPU 时间，不是用 Python time 直接计时。
def benchmark(op, a, b, c=None, warmup=10, rep=100, prefix="torch"):
    input_args = (a, b, c) if c is not None else (a, b)

    # cold_l2_cache=True 选择冷 L2 缓存测量；关闭 CUDA Graph 和 CUPTI 是本实验配置，
    # 不是所有工作负载的最佳选择。比较不同实现时，应保持数据、缓存策略和计时方法一致。
    times = bench_gpu_time(
        fn=op,
        input_args=input_args,
        dry_run_iters=warmup,
        repeat_iters=rep,
        enable_cupti=False,
        use_cuda_graph=False,
        cold_l2_cache=True,
    )
    # 实际统计量是“中位数”，尽管原输出写着 mean time；注释版不改动输出字符串。
    avg_duration = float(np.median(times))

    if prefix == "torch":
        # 每次测自定义实现前先测同一 dtype 的 PyTorch，建立这一轮的参考耗时。
        global baseline
        baseline = avg_duration
        print(f"{prefix:30s} mean time: {avg_duration:8.6f} ms")
    else:
        # 加速比 >1 表示本次测量更快，<1 表示更慢；不是精度指标，也不是跨 GPU 的性能保证。
        speedup = baseline / avg_duration
        print(f"{prefix:30s} mean time: {avg_duration:8.6f} ms, speedup: {speedup:.2f}")


# 逐元素比较后取最大绝对误差；.item() 把标量取回 CPU，会等待相关 GPU 结果就绪。
# 1e-5 是此加法测试原有的严格阈值，不能机械照搬到所有 FP16 归约/矩阵乘法。
# 这里只覆盖给定数据，且 Python -O 会禁用 assert；通过它不等于完成全面边界测试。
def diff_check(a, b, prefix="torch"):
    assert torch.max(torch.abs(a - b)).item() < 1e-5, f"{prefix} result diff"


if __name__ == "__main__":
    # 测试核函数：先 FP32，再 FP16；每轮先建立 PyTorch 结果和计时基线。
    device = torch.device("cuda")
    # 这些尺寸使元素总数对齐到向量组和整块线程；请勿直接换成任意奇数尺寸。
    # 某些核在边界检查前预读整组数据，详见 elementwise.cu 中的原实现限制。
    sz = [1024, 2048, 4096]
    for n in sz:
        for m in sz:
            print("#" * 100)
            print(f"n: {n}, m: {m}")
            a = torch.randn(n, m).float().cuda()
            b = torch.randn(n, m).float().cuda()
            # empty_like 只分配输出空间，不初始化数值；下面 torch.add(out=c) 会写入结果。
            c = torch.empty_like(a).float().cuda()

            # partial 预先绑定 out=c，使参考实现也复用输出空间。
            benchmark(partial(torch.add, out=c), a, b)
            c_my = torch.empty_like(a)
            benchmark(lib.elementwise_add, a, b, c_my, prefix="elementwise_add")
            diff_check(c, c_my, prefix="elementwise_add")
            benchmark(
                lib.elementwise_add_fp32x4,
                a,
                b,
                c_my,
                prefix="elementwise_add_fp32x4",
            )
            diff_check(c, c_my, prefix="elementwise_add_fp32x4")

            # 半精度阶段：.half() 是真正的数值/存储类型转换，与 C++ 指针重解释不同。
            a = a.half()
            b = b.half()
            c = c.half()
            # partial 预先绑定 out=c，使参考实现也复用输出空间。
            benchmark(partial(torch.add, out=c), a, b)
            c_my = c_my.half()
            benchmark(lib.elementwise_add, a, b, c_my, prefix="ele_add_half")
            diff_check(c, c_my, prefix="ele_add_half")
            benchmark(
                lib.elementwise_add_fp16x2,
                a,
                b,
                c_my,
                prefix="elementwise_add_fp16x2",
            )
            diff_check(c, c_my, prefix="elementwise_add_fp16x2")
            benchmark(
                lib.elementwise_add_fp16x8,
                a,
                b,
                c_my,
                prefix="elementwise_add_fp16x8",
            )
            diff_check(c, c_my, prefix="elementwise_add_fp16x8")
            benchmark(
                lib.elementwise_add_fp16x8_packed,
                a,
                b,
                c_my,
                prefix="elementwise_add_fp16x8_packed",
            )
            diff_check(c, c_my, prefix="elementwise_add_fp16x8_packed")
