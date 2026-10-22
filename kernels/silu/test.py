# 中文导读：本脚本保留上游基准与误差检查设置；导入时会触发 CUDA 扩展编译。
import numpy as np
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from flashinfer.testing.utils import bench_gpu_time
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
current_dir = Path(__file__).parent.resolve()
# 将 CUDA 源文件即时编译并加载为 Python 模块；首次运行需要本机 CUDA 工具链。
lib = load(
    name="silu_lib",
    sources=[str(current_dir / "silu.cu")],
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


baseline = None


def benchmark(op, a, b=None, warmup=10, rep=100, prefix="torch"):
    global baseline
    input_args = (a, b) if b is not None else (a,)

    # 计时器预热后重复测量，并请求冷 L2；加速比为先前 torch 基准耗时/当前耗时。
    times = bench_gpu_time(
        fn=op,
        input_args=input_args,
        dry_run_iters=warmup,
        repeat_iters=rep,
        enable_cupti=False,
        use_cuda_graph=False,
        cold_l2_cache=True,
    )
    # 实际取中位数；原日志写 mean time，翻译不改变该可观察字符串。
    avg_duration = float(np.median(times))

    if prefix == "torch":
        baseline = avg_duration
        print(f"{prefix:30s} mean time: {avg_duration:8.6f} ms")
    else:
        speedup = baseline / avg_duration
        print(f"{prefix:30s} mean time: {avg_duration:8.6f} ms, speedup: {speedup:.2f}")


# 检查平均绝对误差；均值达标不能保证每个元素都达标，也不覆盖全部形状或浮点边界。
def diff_check(a, b, prefix="torch", eps=1e-5):
    message = f"{prefix} result diff"
    assert torch.mean(torch.abs(a - b)).item() < eps, message


if __name__ == "__main__":
    # 遍历原始大尺寸测试输入；这些尺寸覆盖整组访存，不代表已经验证任意尾部。
    device = torch.device("cuda")
    sz = [1024, 2048, 4096]
    for n in sz:
        for m in sz:
            print("#" * 100)
            print(f"n: {n}, m: {m}")
            a = torch.randn(n, m).float().cuda()
            b = a.clone()

            # 原地基准会反复修改 b；其分配方式与自定义核输出预分配方式存在差别。
            benchmark(partial(F.silu, inplace=True), b)
            b_my = torch.empty_like(a)

            benchmark(lib.silu, a, b_my, prefix="silu")
            b = F.silu(a)
            diff_check(b, b_my, prefix="silu")
            benchmark(lib.silu_fp32x4, a, b_my, prefix="silu_fp32x4")
            b = F.silu(a)
            diff_check(b, b_my, prefix="silu_fp32x4")

            # 半精度测试：保留上游各版本独立设置的误差阈值。
            a = a.half()
            b = a.clone()
            benchmark(partial(F.silu, inplace=True), b)
            b_my = b_my.half()

            benchmark(lib.silu, a, b_my, prefix="silu_half")
            b = F.silu(a)
            diff_check(b, b_my, prefix="silu_half", eps=1e-3)
            benchmark(lib.silu_fp16x2, a, b_my, prefix="silu_fp16x2")
            b = F.silu(a)
            diff_check(b, b_my, prefix="silu_fp16x2", eps=1e-3)
            benchmark(lib.silu_fp16x8, a, b_my, prefix="silu_fp16x8")
            b = F.silu(a)
            diff_check(b, b_my, prefix="silu_fp16x8", eps=1e-3)
            benchmark(
                lib.silu_fp16x8_packed,
                a,
                b_my,
                prefix="silu_fp16x8_packed",
            )
            b = F.silu(a)
            diff_check(b, b_my, prefix="silu_fp16x8_packed", eps=1e-3)
