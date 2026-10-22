# Flash Decoding（解码阶段注意力）

[英文原文](./readme.en.md) · [中文首页](../../README.md) · [翻译进度](../../docs/zh-CN/翻译进度.md)

> 本页翻译并保留原始章节、命令与历史输出；本次仅修改文档和源码注释，未进行 CUDA 编译、GPU 正确性测试或性能复测。

## 概述

面向 SM120（Blackwell 系列）目标的 Flash Decoding 实现。

- [x] `flash_decode_tma_128`：K/V 单缓冲版本
- [x] `flash_decode_tma_dbf_k`：K 双缓冲版本（Python 实际入口带 `_128` 后缀）
- [x] 算子绑定、与 `torch.compile` 基准对比及结果差异检查

## 中文阅读指引与输入约定

[CUDA 实现](./flash_decode.cu) 针对单个查询位置：Q/O 为 `[q_head,128]`，K/V 为 `[kv_len,kv_head,128]`，没有额外 batch 维。当前入口固定按 BF16 解释张量，内部点积、在线 softmax 和中间工作区使用 FP32。Q 头数必须能被正的 KV 头数整除；原包装未完整校验 dtype、张量形状和同设备关系。

第一阶段按 KV 序列分 chunk，每块用 128 个线程，组成 8 个 16 线程子组。每个线程负责头维的 8 个元素，子组协作计算分数与局部输出；`current_bn` 用来跳过最后一个分块中的无效行。第二阶段按各 chunk 的 log-sum-exp 权重合并，不能简单对局部输出取平均。`get_chunk_size` 使用固定 `num_sms=26` 选择 chunk 大小，属于原实现的调度假设。

单缓冲版本保存一块 K 和一块 V；`_dbf_k` 版本用两个 K 缓冲预取下一块，同时保留一个 V 缓冲。**需要单独审查的原实现限制：**单缓冲 KV 循环末尾没有全块屏障，而下一轮线程 0 会再次向同一共享内存缓冲发起拷贝。所有线程束是否都已读完旧数据，需要同步审查和 GPU 竞态验证；本次没有修复或宣称验证通过。双缓冲版本有循环末尾的 `__syncthreads()`，不能把两者的同步条件混为一谈。

原测试中的 `mean time` 实际取 `np.median(times)`；GB/s 按 Q/K/V/O 元素字节数估算，不是硬件计数器测得的物理总流量。原日志中的 `prefill` 标签按原文保留，本模块计算的是单查询解码。

## 运行测试

```bash
export torch.compile_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

译注：上面的原始命令含笔误，`torch.compile_CUDA_ARCH_LIST` 带有句点，不是合法的 shell 变量名。运行时应使用下面的变量名；这是供 CUDA 环境执行的修正说明，本次未运行：

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

## 历史示例输出

原作者说明：已有许多优秀的 Flash Decoding 核函数针对 SM8x 或企业级 SM100/SM110 优化，但当时较少看到专门面向 SM120 架构、尤其是 RTX 50 系列消费级显卡的底层极致优化。

为追求峰值性能，原作者实现了专用的 split-K 核函数。核心设计是不对称 TMA 缓冲：直接使用 TMA 异步拷贝，并为 K 配置双缓冲，目标是在 $Q \times K^T$ 阶段隐藏内存访问延迟。

V 则有意保留单缓冲。原作者通过这一取舍节省共享内存（SMEM），以改善消费级 GPU 上的驻留与调度条件，并尽量利用内存带宽；下面是当时记录的结果。

原作者报告：与 `torch.compile` 和 FlashInfer 对比时，该实现的带宽可达到约 377 GB/s，并展示了非整块序列长度（例如 `seq=131073`）的结果。原文将其描述为接近物理带宽上限；这属于原始性能叙述，不是本次复测结论，也不能据此保证任意不对齐输入都正确。

原始基准环境：RTX 5060，头数 32，头维 128。
```yaml
####################################################################################################
prefill, kv seq: 8192, head: 32, dim: 128
torch.compile                            mean time: 0.454655 ms, 295.24 GB/s
flash-infer                              mean time: 0.403291 ms, speedup: 1.13, GB/s: 332.85
flash_decode_tma_128                     mean time: 0.408378 ms, speedup: 1.11, GB/s: 328.70
flash_decode_tma_dbf_k_128               mean time: 0.366698 ms, speedup: 1.24, GB/s: 366.06
####################################################################################################
prefill, kv seq: 16384, head: 32, dim: 128
torch.compile                            mean time: 0.872882 ms, 307.55 GB/s
flash-infer                              mean time: 0.784423 ms, speedup: 1.11, GB/s: 342.23
flash_decode_tma_128                     mean time: 0.735274 ms, speedup: 1.19, GB/s: 365.10
flash_decode_tma_dbf_k_128               mean time: 0.733273 ms, speedup: 1.19, GB/s: 366.10
####################################################################################################
prefill, kv seq: 32768, head: 32, dim: 128
torch.compile                            mean time: 1.507921 ms, 356.04 GB/s
flash-infer                              mean time: 1.499479 ms, speedup: 1.01, GB/s: 358.05
flash_decode_tma_128                     mean time: 1.495797 ms, speedup: 1.01, GB/s: 358.93
flash_decode_tma_dbf_k_128               mean time: 1.455790 ms, speedup: 1.04, GB/s: 368.79
####################################################################################################
prefill, kv seq: 65536, head: 32, dim: 128
torch.compile                            mean time: 2.980080 ms, 360.31 GB/s
flash-infer                              mean time: 2.897006 ms, speedup: 1.03, GB/s: 370.64
flash_decode_tma_128                     mean time: 2.856871 ms, speedup: 1.04, GB/s: 375.85
flash_decode_tma_dbf_k_128               mean time: 2.849400 ms, speedup: 1.05, GB/s: 376.84
####################################################################################################
prefill, kv seq: 131072, head: 32, dim: 128
torch.compile                            mean time: 6.044398 ms, 355.29 GB/s
flash-infer                              mean time: 5.751600 ms, speedup: 1.05, GB/s: 373.37
flash_decode_tma_128                     mean time: 5.663495 ms, speedup: 1.07, GB/s: 379.18
flash_decode_tma_dbf_k_128               mean time: 5.736955 ms, speedup: 1.05, GB/s: 374.33
####################################################################################################
prefill, kv seq: 131073, head: 32, dim: 128
torch.compile                            mean time: 6.466227 ms, 332.11 GB/s
flash-infer                              mean time: 6.117131 ms, speedup: 1.06, GB/s: 351.07
flash_decode_tma_128                     mean time: 5.701174 ms, speedup: 1.13, GB/s: 376.68
flash_decode_tma_dbf_k_128               mean time: 5.695415 ms, speedup: 1.14, GB/s: 377.06
```
