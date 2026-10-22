# Flash Attention（分块注意力）

[英文原文](./readme.en.md) · [中文首页](../../README.md) · [翻译进度](../../docs/zh-CN/翻译进度.md)

> 本页翻译并保留原始章节、命令与历史输出；本次仅修改文档和源码注释，未进行 CUDA 编译、GPU 正确性测试或性能复测。

## 概述

面向 SM120（Blackwell 系列）目标的 Flash Attention 实现。

- [x] `fmha_tma_128`：单缓冲，BM×BN = 64×64；原作者面向较短序列选择的分块配置
- [x] PyTorch 算子绑定与结果差异检查

实现中的主要技术选择如下：

- BM×BN = 64×64 分块：针对 `head_dim=128` 的序列分块设计。
- 直接使用 PTX 的 TMA：把头维拆成两部分，每个分块发起两次 TMA 拷贝。
- 延迟重缩放：积累行缩放系数，意在减少在线 softmax 中反复缩放输出寄存器的开销。

## 中文阅读指引与输入约定

从 [CUDA 实现](./flash_attn.cu) 的 `fmha_tma_kernel` 主循环开始阅读，再看矩阵片段加载、`lazy_rescale` 和主机包装。Q/O 使用 `[batch,q_len,q_head,128]`，K/V 使用 `[batch,kv_len,kv_head,128]`；当前绑定直接按 BF16 解释所有指针，点积与输出累加器使用 FP32。

一个线程块处理某个 batch、某个 Q 头的 64 行查询，四个线程束各负责 16 行。Q 加载后在 KV 循环中复用；K/V 每次加载 64 行。在线 softmax 同时维护行最大值与分母，最大值变化时要修正已有分子和分母的尺度。此实现把部分修正延迟到 `row_scale`，在尺度过小时才落实；最终输出仍要除以相应分母。

因果掩码直接比较 Q 与 K 的位置编号，不包含已有 KV 缓存的额外位置偏移。原测试使用相同 Q/KV 长度、32 个头、头维 128；这些配置不等于完整的变长或分组注意力验证。调用方须保证 `kv_head>0`、`q_head` 可被 `kv_head` 整除，并满足 BF16、同设备、连续、各张量形状和输出空间匹配。`CHECK_T` 没有完成全部检查。

当前没有为最后一个 KV 分块额外生成 `k_position<kv_len` 的有效长度掩码。当 `q_len>kv_len` 且 `kv_len` 不是 64 的倍数时，无效键可能通过有效 Q 行的因果判断并参与 softmax。例如 `q_len=2、kv_len=1` 时，第二个 Q 位置会允许并不存在的 K 列 1。这个纸面例子说明，不能把 TMA 的越界填充当作无效键掩码，也不能把原测试扩展解释为任意矩形长度均受支持。

TMA、共享内存重排和屏障阶段必须配合使用，不能只改分块常量就假定正确。原 `mbarrier.try_wait` 注释把等待提示写成时钟周期；依据 [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)，该参数单位为纳秒，且只是单次等待提示，本分支已在注释中澄清。

## 运行测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

## 历史示例输出

```yaml
####################################################################################################
prefill, batch:  1, seq: 512, head: 32, dim: 128
torch                                    mean time: 0.121095 ms, 17.73 tflops
fmha_tma_128                             mean time: 0.098975 ms, speedup: 1.22, tflops: 21.70
####################################################################################################
prefill, batch:  1, seq: 1024, head: 32, dim: 128
torch                                    mean time: 0.399812 ms, 21.48 tflops
fmha_tma_128                             mean time: 0.352873 ms, speedup: 1.13, tflops: 24.34
####################################################################################################
prefill, batch:  1, seq: 2048, head: 32, dim: 128
torch                                    mean time: 1.355930 ms, 25.34 tflops
fmha_tma_128                             mean time: 1.293091 ms, speedup: 1.05, tflops: 26.57
####################################################################################################
prefill, batch:  1, seq: 4096, head: 32, dim: 128
torch                                    mean time: 4.983902 ms, 27.58 tflops
fmha_tma_128                             mean time: 4.891181 ms, speedup: 1.02, tflops: 28.10
####################################################################################################
prefill, batch:  1, seq: 8192, head: 32, dim: 128
torch                                    mean time: 18.248972 ms, 30.13 tflops
fmha_tma_128                             mean time: 17.612578 ms, speedup: 1.04, tflops: 31.21
```
