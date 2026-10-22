# 中文阅读提示：此脚本会即时编译 CUDA 扩展；下面是原有测试配置，本次未执行 GPU 测试。
# 本入口供 Nsight Compute 采集使用；分析器运行会改变计时条件，不能直接与普通计时混用。
# 当前正余弦缓存只预建 head_dim=128/256、位置 0..8191；更大位置或其他维度需另行设计。
# rope() 中现算的 cos/sin 随后被全局缓存覆盖，这是原实现的行为。
import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
# 将 CUDA 源码即时编译并加载为 Python 扩展模块。
lib = load(
    name="rope_lib",
    sources=["rope_neox.cu"],
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
base = 10000


def compute_default_rope_parameters(head_dim):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2).float().cuda() / head_dim)
    )  # 64
    return inv_freq


INV_FREQS = {
    256: compute_default_rope_parameters(256),
    128: compute_default_rope_parameters(128),
}
position_ids = torch.arange(8192).float().cuda()
freqs = {
    256:torch.cat([torch.outer(position_ids, INV_FREQS[256]), torch.outer(position_ids, INV_FREQS[256])], dim=-1),
    128:torch.cat([torch.outer(position_ids, INV_FREQS[128]), torch.outer(position_ids, INV_FREQS[128])], dim=-1),
}
COS = {
    256:torch.cos(freqs[256]),
    128:torch.cos(freqs[128]),
}
SIN = {
    256:torch.sin(freqs[256]),
    128:torch.sin(freqs[128]),
}

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed
#@torch.compile()
def rope_with_sin_cos_cache(q):  # q 的形状：[bs, seqlen, head_dim]。
    # inv_freq = compute_default_rope_parameters(q.shape[-1])
    # position_ids = torch.arange(q.shape[1], device=q.device).float()

    # # 位置与逆频率做外积：[seq_len] × [dim/2] -> [seq_len, dim/2]。
    # freqs = torch.outer(position_ids, inv_freq)

    # # [seq_len, dim/2] -> [seq_len, dim]
    # freqs = torch.cat([freqs, freqs], dim=-1)

    # cos, sin = torch.cos(freqs), torch.sin(freqs)
    cos = COS[q.shape[-1]][:q.shape[1], :q.shape[-1]]
    sin = SIN[q.shape[-1]][:q.shape[1], :q.shape[-1]]

    return apply_rotary_pos_emb(q, cos, sin)

# NeoX 风格的 RoPE；当前函数处理 [bs, seqlen, head_dim]，原注释的“单 batch”描述不准确。
#@torch.compile()
def rope(q):  # q 的形状：[bs, seqlen, head_dim]。
    inv_freq = compute_default_rope_parameters(q.shape[-1])
    position_ids = torch.arange(q.shape[1], device=q.device).float()

    # 位置与逆频率做外积：[seq_len] × [dim/2] -> [seq_len, dim/2]。
    freqs = torch.outer(position_ids, inv_freq)

    # [seq_len, dim/2] -> [seq_len, dim]
    freqs = torch.cat([freqs, freqs], dim=-1)

    cos, sin = torch.cos(freqs), torch.sin(freqs)
    cos = COS[q.shape[-1]][:q.shape[1], :q.shape[-1]]
    sin = SIN[q.shape[-1]][:q.shape[1], :q.shape[-1]]

    return apply_rotary_pos_emb(q, cos, sin)


def benchmark(op, a, b=None, warmup=0, rep=1, prefix="torch"):
    if b is not None:
        # 预热，减少首次运行对正式计时的影响。
        for i in range(warmup):
            res = op(a, b)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(rep):
            res = op(a, b)
        torch.cuda.synchronize()
        print(f"{prefix:30s} mean time: {(time.perf_counter() - start) / rep * 1000:.6f} ms")
    else:
        # 预热，减少首次运行对正式计时的影响。
        for i in range(warmup):
            res = op(a)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(rep):
            res = op(a)
        torch.cuda.synchronize()
        print(f"{prefix:30s} mean time: {(time.perf_counter() - start) / rep * 1000:.6f} ms")
    return res


def diff_check(a, b, prefix="torch", eps=1e-4):
    message = f"{prefix} result diff"
    assert torch.mean(torch.abs(a - b)).item() < eps, message


if __name__ == "__main__":
    # 运行下面列出的核函数测试配置。
    device = torch.device("cuda")

    torch.manual_seed(42)
    bs, n, m = 128, 8192, 128
    print("#" * 100)
    print(f"bs: {bs}, n: {n}, m: {m}")
    a = torch.randn(bs, n, m).float().cuda()

    b = benchmark(rope, a)
    c = benchmark(rope_with_sin_cos_cache, a, prefix="torch.rope_with_sin_cos_cache")
    diff_check(b, c, prefix="rope_with_sin_cos_cache")
    b_my = torch.empty_like(a)
    benchmark(lib.rope, a, b_my, prefix="rope")

    benchmark(lib.rope_fp32x4, a, b_my, prefix="rope_fp32x4")
