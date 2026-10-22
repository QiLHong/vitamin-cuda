# 中文导读：比较不同转置访存策略；输出显式分配为输入的行列互换形状。
# 当前尺寸都是整齐分块，不能由这些用例推断 packed 版本支持任意非整组尾部。
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
    name="transpose_lib",
    sources=[str(current_dir / "transpose.cu")],
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

# @torch.compile()
def transpose(a, b):
    return torch.transpose_copy(a, dim0=0, dim1=1, out=b)

baseline = None

def benchmark(op, a, b=None, warmup=10, rep=100, prefix="torch"):
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


def diff_check(a, b, prefix="torch", eps=1e-6):
    if not torch.allclose(a, b, atol=eps, rtol=eps):
        print(f"{prefix} result diff: {torch.mean(torch.abs(a - b)).item()}")
    assert torch.allclose(a, b, atol=eps, rtol=eps), "result diff"


if __name__ == "__main__":
    # 运行原有测试尺寸与基准流程。
    device = torch.device("cuda")
    sz = [256, 512, 1024, 2048, 4096, 8192]
    torch.manual_seed(42)
    for n in sz:
        for m in sz:
            print("#" * 100)
            print(f"n: {n}, m: {m}")
            a = torch.arange(n * m).reshape(n, m).contiguous().float().cuda()
            b = torch.randn(m, n).float().cuda()

            benchmark(transpose, a, b)
            b_my = torch.zeros_like(b)
            benchmark(lib.transpose_coalesced_read, a, b_my, prefix="transpose_coalesced_read")
            # print(b, b_my)
            diff_check(b, b_my, prefix="transpose_coalesced_read")
            benchmark(lib.transpose_coalesced_write, a, b_my, prefix="transpose_coalesced_write")
            diff_check(b, b_my, prefix="transpose_coalesced_write")

            benchmark(lib.transpose_smem, a, b_my, prefix="transpose_smem")
            diff_check(b, b_my, prefix="transpose_smem")
            benchmark(lib.transpose_smem_bcf, a, b_my, prefix="transpose_smem_bcf")
            diff_check(b, b_my, prefix="transpose_smem_bcf")
            benchmark(lib.transpose_smem_packed_bcf, a, b_my, prefix="transpose_smem_packed_bcf")
            diff_check(b, b_my, prefix="transpose_smem_packed_bcf")
            benchmark(lib.transpose_smem_swizzled_packed, a, b_my, prefix="transpose_smem_swizzled_packed")
            diff_check(b, b_my, prefix="transpose_smem_swizzled_bcf_packed")
