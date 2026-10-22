# 中文导读：比较 PyTorch 点积和标量/打包 CUDA 入口，绝对误差或相对误差满足其一即通过。
# 半精度乘法、返回舍入和原子加分组可能产生数值差异；原容差与输入范围保持不变。
# 本次只翻译和补充注释，未执行 CUDA 编译、GPU 正确性或性能测试。
import numpy as np
from functools import partial
from pathlib import Path

import torch
from flashinfer.testing.utils import bench_gpu_time
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
current_dir = Path(__file__).parent.resolve()
# 即时编译 CUDA 源文件并加载为 Python 扩展；首次执行需要可用的 CUDA 编译环境。
lib = load(
    name="dot_product_lib",
    sources=[str(current_dir / "dot_product.cu")],
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

def benchmark(op, a, b, warmup=10, rep=100, prefix="torch"):
    global baseline

    # 使用原设置预热、重复计时并启用 cold_l2_cache；不在本次翻译中改变测量条件。
    times = bench_gpu_time(
        fn=op,
        input_args=(a, b),
        dry_run_iters=warmup,
        repeat_iters=rep,
        enable_cupti=False,
        use_cuda_graph=False,
        cold_l2_cache=True,
    )
    # 实际统计量是中位数；输出中的英文 mean time 标签按原样保留。
    avg_duration = float(np.median(times))

    if prefix == "torch":
        baseline = avg_duration
        print(f"{prefix:30s} mean time: {avg_duration:8.6f} ms")
    else:
        speedup = baseline / avg_duration
        print(f"{prefix:30s} mean time: {avg_duration:8.6f} ms, speedup: {speedup:.2f}")

    return op(a, b)


def diff_check(a, b, prefix="torch", rel=1e-4, abl=1e-4):
    message = f"{prefix} result diff"
    assert (
        torch.max(torch.abs(a - b)).item() < abl
        or torch.max(torch.abs(a - b) / torch.abs(a)).item() < rel
    ), message


if __name__ == "__main__":
    # 运行原有测试尺寸与基准流程。
    device = torch.device("cuda")
    sz = [1024, 2048, 4096]
    for n in sz:
        for m in sz:
            print("#" * 100)
            print(f"n: {n}, m: {m}")
            a = torch.randn(n * m).float().cuda()
            b = torch.randn(n * m).float().cuda()

            c = benchmark(torch.dot, a, b)
            c_my = benchmark(lib.dot_product, a, b, prefix="dot_product")
            # print(c, c_my)
            diff_check(c, c_my, prefix="dot_product")
            c_my = benchmark(
                lib.dot_product_fp32x4,
                a,
                b,
                prefix="dot_product_fp32x4",
            )
            diff_check(c, c_my, prefix="dot_product_fp32x4")

            ################### 半精度（FP16）
            a = a.half()
            b = b.half()
            c = benchmark(torch.dot, a, b)
            c_my = benchmark(lib.dot_product, a, b, prefix="dot_product")
            # print(c, c_my)
            diff_check(c, c_my, prefix="dot_product", abl=1e-2, rel=1e-2)
            c_my = benchmark(
                lib.dot_product_fp16x2,
                a,
                b,
                prefix="dot_product_fp16x2",
            )
            diff_check(c, c_my, prefix="dot_product_fp16x2", abl=1e-2, rel=1e-2)
            c_my = benchmark(
                lib.dot_product_fp16x8_packed,
                a,
                b,
                prefix="dot_product_fp16x8_packed",
            )
            diff_check(c, c_my, prefix="dot_product_fp16x8_packed", abl=1e-2, rel=1e-2)
