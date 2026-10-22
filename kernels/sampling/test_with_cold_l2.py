"""Isolate the L2 effect: identical timer (CUDA events, no graph, no CUPTI), toggling ONLY cold_l2_cache (False=warm/resident, True=flushed each iter)."""

# 中文阅读提示：此脚本会即时编译 CUDA 扩展；下面是原有测试配置，本次未执行 GPU 测试。
# 计时并不验证采样分布；不同实现相同 seed 也不自动保证逐词元相同。
# 当前 CUDA 源码的 BF16 路径存在按 FP16 解码的问题，详见 sampling.cu 与中文 README。

import numpy as np, torch
import test as bench  # 导入时触发自定义 CUDA 扩展的即时编译。
import flashinfer
from flashinfer.testing.utils import bench_gpu_time

torch.set_grad_enabled(False)
lib = bench.lib
TOPK, TOPP, SEED, OFF = 20, 0.95, 42, 1


def gen_spiked(bs, V):
    return bench.generate_realistic_logits(bs, V, num_spikes=50)


# 原始模块 docstring 的中文说明：固定 CUDA Event 计时器，不用 Graph/CUPTI，
# 只切换 cold_l2_cache；False 不要求每轮清缓存，True 请求每轮清缓存。
# 返回计时样本中位数；“warm”标签本身不是缓存命中率已实测的证明。
def measure(fn, logits, cold):
    times = bench_gpu_time(
        fn=fn,
        input_args=(logits,),
        dry_run_iters=10,
        repeat_iters=100,
        enable_cupti=False,
        use_cuda_graph=False,  # 仅使用 CUDA Event 计时，不使用 CUDA Graph。
        cold_l2_cache=cold,
    )
    return float(np.median(times))


fns = {
    "flashinfer": lambda l: flashinfer.sampling.top_k_top_p_sampling_from_logits(
        l, TOPK, TOPP, filter_apply_order="top_k_first", seed=SEED, offset=OFF
    ),
    "custom_batched": lambda l: lib.sampling_topk_topp_batched(
        l, TOPK, TOPP, SEED, OFF
    ),
    "custom_splitk": lambda l: lib.sampling_topk_topp_split_k(l, TOPK, TOPP, SEED, OFF),
}

ns = [1, 4, 8, 16, 32]
ms = [128000, 256000, 320000]
configs = []
for bs in ns:
    for vocab_size in ms:
        configs.append((bs, vocab_size))
print(
    f"GPU: {torch.cuda.get_device_name(0)}  fi={getattr(flashinfer,'__version__','?')}"
)
print("timer = CUDA events, no graph, no CUPTI; only cold_l2_cache toggled\n")
print(
    f"{'config':<14}{'kernel':<16}{'warmL2(cold=F)':>16}{'coldL2(cold=T)':>16}{'cold/warm':>11}{'cold speedup':>16}"
)
for bs, V in configs:
    print("#" * 100)
    sp = gen_spiked(bs, V)

    fi_warm = measure(fns["flashinfer"], sp, cold=False)
    fi_cold = measure(fns["flashinfer"], sp, cold=True)
    fi_baseline = fi_cold
    print(
        f"{f'bs{bs}/{V//1000}k':<14}{'flashinfer':<16}{fi_warm:>16.4f}{fi_cold:>16.4f}{fi_cold/fi_warm:>11.2f}x"
    )

    for name, fn in fns.items():
        if name == "flashinfer":
            continue
        if name == "custom_splitk" and bs > 8:
            continue

        warm = measure(fn, sp, cold=False)
        cold = measure(fn, sp, cold=True)
        speedup = fi_baseline / cold

        print(
            f"{f'bs{bs}/{V//1000}k':<14}{name:<16}{warm:>16.4f}{cold:>16.4f}{cold/warm:>11.2f}x{speedup:>11.2f}x"
        )
