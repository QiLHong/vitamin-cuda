# 中文导读：并列比较 Triton、TileLang 与手写 CUDA 加法，保留全部基准参数。
# 导入 kernels.elementwise.test 会编译 CUDA 扩展；从本目录执行以匹配相对搜索路径。
import sys
from functools import partial
import torch
import tilelang
import tilelang.language as T

import triton
import triton.language as tl
from flashinfer.testing.utils import bench_gpu_time
import numpy as np

sys.path.append("../../../")
from kernels.elementwise.test import lib


# Triton 对照：完整块直接读写，末块使用 offsets<n 掩码。
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    st = pid * BLOCK_SIZE
    offsets = st + tl.arange(0, BLOCK_SIZE)

    if st + BLOCK_SIZE <= n:
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        out = x + y
        tl.store(out_ptr + offsets, out)
    else:
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        out = x + y
        tl.store(out_ptr + offsets, out, mask=mask)


# TileLang 基础版：把每个元素的加法交给 T.Parallel 映射到线程。
@tilelang.jit
def add_tilelang(N: int, block: int = 256, dtype: str = "float16"):

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                out_id = bx * block + i
                C[out_id] = A[out_id] + B[out_id]

    return main


# 显式向量化：每线程负责 vec=8 个连续元素，完整组与尾部分别走不同循环。
# 对 base>=N 的线程，尾部循环的处理取决于 DSL 生成结果，需要在实际工具链检查。
@tilelang.jit
def add_tilelang_vectorized(N: int, block: int = 256, dtype: str = "float16"):
    vec = 8

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        grid = T.ceildiv(N, block * vec)

        with T.Kernel(grid, threads=block) as bx:
            tid = T.get_thread_binding(0)
            tile_id = bx * block + tid
            base = tile_id * vec

            if base + 7 < N:
                for i in T.vectorized(vec):
                    elem = base + i
                    C[elem] = A[elem] + B[elem]
            else:
                for i in T.serial(N - base):
                    elem = base + i
                    C[elem] = A[elem] + B[elem]

    return main


# 即时调用形式：N 从输入推断。vec 按位宽计算，但完整组判断仍硬编码 base+7。
# 当前演示使用 float16；不要把可选 dtype 参数理解为所有位宽均已验证。
@tilelang.jit
def add_tilelang_vectorized_eager(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dtype = "float16"
):
    N = T.const("N")
    block = 256
    vec = 128 // tilelang.DataType(dtype).bits
    A: T.Tensor((N,), dtype)
    B: T.Tensor((N,), dtype)
    C: T.Tensor((N,), dtype)

    with T.Kernel(T.ceildiv(N, block * vec), threads=block) as bx:
        tid = T.get_thread_binding(0)
        tile_id = bx * block + tid
        base = tile_id * vec

        if base + 7 < N:
            for i in T.vectorized(vec):
                elem = base + i
                C[elem] = A[elem] + B[elem]
        else:
            for i in T.serial(N - base):
                elem = base + i
                C[elem] = A[elem] + B[elem]

# 共享内存版本先搬运两份输入，再经 fragment 计算并写回。
# 本实验用于比较 DSL 生成策略，不意味着简单加法天然需要共享内存。
@tilelang.jit
def elementwise_add(A, B, C, block_N=2048, dtype="float16", threads=256):
    N = T.const("N")

    A: T.Tensor((N), dtype)
    B: T.Tensor((N), dtype)
    C: T.Tensor((N,), dtype)

    with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
        A_shared = T.alloc_shared((block_N), dtype)
        B_shared = T.alloc_shared((block_N), dtype)
        C_local = T.alloc_fragment((block_N), dtype)
        C_shared = T.alloc_shared((block_N), dtype)

        T.copy(A[bx * block_N], A_shared)
        T.copy(B[bx * block_N], B_shared)
        for local_x in T.Parallel(block_N):
            C_local[local_x] = A_shared[local_x] + B_shared[local_x]
        T.copy(C_local, C_shared)
        T.copy(C_shared, C[bx * block_N])

    # return C

# 去掉显式共享内存：虽然变量仍叫 A_shared/B_shared，实际分配的是 fragment。
@tilelang.jit
def elementwise_add_no_shared(A, B, C, block_N=2048, dtype="float16", threads=256):
    N = T.const("N")

    A: T.Tensor((N), dtype)
    B: T.Tensor((N), dtype)
    C: T.Tensor((N,), dtype)

    with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
        A_shared = T.alloc_fragment((block_N), dtype)
        B_shared = T.alloc_fragment((block_N), dtype)
        C_local = T.alloc_fragment((block_N), dtype)

        T.copy(A[bx * block_N], A_shared)
        T.copy(B[bx * block_N], B_shared)
        for local_x in T.Parallel(block_N):
            C_local[local_x] = A_shared[local_x] + B_shared[local_x]
        T.copy(C_local, C[bx * block_N])

def diff_check(a, b, prefix="torch", eps=1e-3):
    if not torch.allclose(a, b, atol=eps, rtol=eps):
        diff = torch.abs(a - b)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        print(f"{prefix} result mean diff: {mean_diff:.6f}, max diff: {max_diff:.6f}")
    assert torch.allclose(a, b, atol=eps, rtol=eps), "result diff"


def benchmark(op, x, y, o=None, prefix="torch"):
    def measure(fn, x, y, o, cold):
        times = bench_gpu_time(
            fn=fn,
            input_args=(x, y, o) if o is not None else (x, y),
            dry_run_iters=10,
            repeat_iters=100,
            enable_cupti=False,
            use_cuda_graph=False,  # 使用 CUDA 事件计时，不使用 CUDA Graph
            cold_l2_cache=cold,
        )
        # 统计中位数；下面原输出字符串写 mean time，保持原样以方便与上游对照。
        return float(np.median(times))

    avg_duration = measure(op, x, y, o, cold=True)
    if prefix == "torch":
        global baseline
        baseline = avg_duration
        print(f"{prefix:40s} mean time: {avg_duration:8.6f} ms")
    else:
        speedup = baseline / avg_duration
        print(
            f"{prefix:40s} mean time: {avg_duration:8.6f} ms, speedup: {speedup:.2f}"
        )
    return o


def test_all():
    torch.manual_seed(42)
    DEVICE = torch.device("cuda")
    for n in [1024, 4096, 1024 * 32, 1024 * 1024, 1024 * 4096, 4096 * 4096]:
        print("#" * 100)
        print(f"vector add, n: {n}")
        x = torch.randn(n, dtype=torch.float32, device=DEVICE)
        y = torch.randn(n, dtype=torch.float32, device=DEVICE)
        out = torch.empty_like(x)

        benchmark(partial(torch.add, out=out), x, y)

        out_my = torch.zeros_like(out)

        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        benchmark(
            partial(add_kernel[grid], n=n, BLOCK_SIZE=256),
            x,
            y,
            out_my,
            prefix="triton",
        )
        diff_check(out, out_my)
        kernel = add_tilelang(n, dtype="float32")
        benchmark(kernel, x, y, out_my, prefix="tilelang")
        benchmark(
            lib.elementwise_add_fp32x4,
            x,
            y,
            out_my,
            prefix="elementwise_add_fp32x4",
        )
        diff_check(out, out_my, prefix="elementwise_add_fp32x4")

        x = x.half()
        y = y.half()
        out = out.half()
        out_my = out_my.half()
        benchmark(partial(torch.add, out=out), x, y)
        benchmark(
            partial(add_kernel[grid], n=n, BLOCK_SIZE=256),
            x,
            y,
            out_my,
            prefix="triton",
        )
        kernel = add_tilelang(n, dtype="float16")
        # print(kernel.get_kernel_source())
        out_my = torch.zeros_like(out_my)
        benchmark(kernel, x, y, out_my, prefix="tilelang")

        kernel = add_tilelang_vectorized(n, dtype="float16")
        # print(kernel.get_kernel_source())
        out_my = torch.zeros_like(out_my)
        benchmark(kernel, x, y, out_my, prefix="tilelang_vectorized")
        # print(out[-1], out_my[-1])
        diff_check(out, out_my)
        out_my = torch.zeros_like(out_my)
        benchmark(
            add_tilelang_vectorized_eager,
            x,
            y,
            out_my,
            prefix="tilelang_vectorized_eager",
        )
        # print(add_tilelang_vectorized_eager.compile(N=n, A=x).get_kernel_source())
        # print(out[-1], out_my[-1])
        diff_check(out, out_my)
        benchmark(
            lib.elementwise_add_fp16x8_packed,
            x,
            y,
            out_my,
            prefix="elementwise_add_fp16x8_packed",
        )
        diff_check(out, out_my, prefix="elementwise_add_fp16x8_packed")


# 专门比较整组长度以及前后各差一个元素的长度；通过情况仍以实际运行结果为准。
def test_sp():
    torch.manual_seed(42)
    DEVICE = torch.device("cuda")
    for n in [4096 * 4096, 4096 * 4096 - 1, 4096 * 4096 + 1]:
        print("#" * 100)
        print(f"vector add, n: {n}")
        x = torch.randn(n, dtype=torch.float32, device=DEVICE)
        y = torch.randn(n, dtype=torch.float32, device=DEVICE)
        out = torch.empty_like(x)

        out_my = torch.zeros_like(out)

        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

        x = x.half()
        y = y.half()
        out = out.half()
        out_my = out_my.half()
        benchmark(partial(torch.add, out=out), x, y)
        benchmark(
            partial(add_kernel[grid], n=n, BLOCK_SIZE=1024),
            x,
            y,
            out_my,
            prefix="triton",
        )
        kernel = add_tilelang(n, dtype="float16")
        # print(kernel.get_kernel_source())
        out_my = torch.zeros_like(out_my)
        benchmark(kernel, x, y, out_my, prefix="tilelang")

        kernel = add_tilelang_vectorized(n, dtype="float16")
        # print(kernel.get_kernel_source())
        out_my = torch.zeros_like(out_my)
        benchmark(kernel, x, y, out_my, prefix="tilelang_vectorized")
        # print(out[-1], out_my[-1])
        diff_check(out, out_my)
        out_my = torch.zeros_like(out_my)
        benchmark(
            add_tilelang_vectorized_eager,
            x,
            y,
            out_my,
            prefix="tilelang_vectorized_eager",
        )
        # print(add_tilelang_vectorized_eager.compile(N=n, A=x).get_kernel_source())
        # print(out[-1], out_my[-1])
        if n%4096 == 0:
            out_my = torch.zeros_like(out_my)
            benchmark(
                elementwise_add,
                x,
                y,
                out_my,
                prefix="elementwise_add",
            )
            # print(elementwise_add.compile(N=n).get_kernel_source())
            diff_check(out, out_my)
        out_my = torch.zeros_like(out_my)
        benchmark(
            elementwise_add_no_shared,
            x,
            y,
            out_my,
            prefix="elementwise_add_no_shared",
        )
        # print(elementwise_add_no_shared.compile(N=n).get_kernel_source())
        diff_check(out, out_my)
        benchmark(
            lib.elementwise_add_fp16x8_packed,
            x,
            y,
            out_my,
            prefix="elementwise_add_fp16x8_packed",
        )
        diff_check(out, out_my, prefix="elementwise_add_fp16x8_packed")


# 原入口只运行 test_sp；test_all 保留为可选入口，未在翻译中改变默认执行范围。
if __name__ == "__main__":
    # test_all()
    test_sp()
