# 中文导读：比较带权重的均方根归一化；原 diff_check 检查平均绝对误差。
# 平均误差阈值不等于每个元素都满足该阈值，原测试范围和判定保持不变。
# 本次只翻译和补充注释，未执行 CUDA 编译、GPU 正确性或性能测试。
from functools import partial
from pathlib import Path

import numpy as np
import torch
from flashinfer.testing.utils import bench_gpu_time, get_l2_cache_size
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
current_dir = Path(__file__).parent.resolve()
# 即时编译 CUDA 源文件并加载为 Python 扩展；首次执行需要可用的 CUDA 编译环境。
lib = load(
    name="rmsnorm_lib",
    sources=[str(current_dir / "rmsnorm.cu")],
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


def rmsnorm(x, weight, dim=1, eps=1e-6, out=None):  # 输入形状为 [n, dim]，权重形状为 [dim]。
    out = x * torch.rsqrt(x.pow(2).mean(dim, keepdim=True) + eps) * weight
    return out


baseline = None


def benchmark(op, a, b, warmup=10, rep=100, prefix="torch"):
    input_args = (a, b)

    # 使用原设置预热、重复计时并启用 cold_l2_cache；不在本次翻译中改变测量条件。
    times = bench_gpu_time(
        fn=op,
        input_args=input_args,
        dry_run_iters=warmup,
        repeat_iters=rep,
        enable_cupti=False,
        use_cuda_graph=False,
        cold_l2_cache=True,
    )
    # 实际统计量是中位数；输出中的英文 mean time 标签按原样保留。
    avg_duration = float(np.median(times))

    if prefix == "torch":
        global baseline
        baseline = avg_duration
        print(f"{prefix:30s} mean time: {avg_duration:8.6f} ms")
    else:
        speedup = baseline / avg_duration
        print(f"{prefix:30s} mean time: {avg_duration:8.6f} ms, speedup: {speedup:.2f}")

    return op(a, b)


def diff_check(a, b, prefix="torch", eps=1e-4):
    message = f"{prefix} result diff"
    assert torch.mean(torch.abs(a - b)).item() < eps, message


if __name__ == "__main__":
    # 运行原有测试尺寸与基准流程。
    device = torch.device("cuda")
    bs = [256, 1024, 2048, 4096]
    sz = [2048, 4096, 8192]
    torch.manual_seed(42)
    for n in bs:
        for m in sz:
            print("#" * 100)
            print(f"n: {n}, m: {m}")
            a = torch.randn(n, m).float().cuda()
            b = torch.randn(m).float().cuda()

            c = benchmark(rmsnorm, a, b)
            c_my = benchmark(lib.rmsnorm, a, b, prefix="rmsnorm")

            diff_check(c, c_my, prefix="rmsnorm")

            c_my = benchmark(lib.rmsnorm_fp32x4, a, b, prefix="rmsnorm_fp32x4")
            diff_check(c, c_my, prefix="rmsnorm_fp32x4")

            c_my = benchmark(lib.rmsnorm_fp32x4_smem, a, b, prefix="rmsnorm_fp32x4_smem")
            diff_check(c, c_my, prefix="rmsnorm_fp32x4")

            a = a.half()
            b = b.half()
            c = benchmark(rmsnorm, a, b)
            c_my = benchmark(lib.rmsnorm, a, b, prefix="rmsnorm_fp16")
            diff_check(c, c_my, prefix="rmsnorm_fp16", eps=1e-3)

            c_my = benchmark(
                lib.rmsnorm_fp16x8_packed, a, b, prefix="rmsnorm_fp16x8_packed"
            )
            diff_check(c, c_my, prefix="rmsnorm_fp16x8_packed", eps=1e-3)

            c_my = benchmark(
                lib.rmsnorm_fp16x8_packed_smem, a, b, prefix="rmsnorm_fp16x8_packed_smem"
            )
            # print(c, c_my)
            diff_check(c, c_my, prefix="rmsnorm_fp16x8_packed_smem", eps=1e-3)
