# Elu

[中文首页](../../README.md) · [术语表](../../docs/zh-CN/术语表.md) · [英文原文](./readme.en.md)

## 功能概览

ELU 激活函数核函数。

- [x] elu — FP32 / FP16
- [x] elu_fp32x4 — FP32 四元素向量化（与源码入口对应）
- [x] elu_fp16x2 — FP16 向量化
- [x] elu_fp16x8 — FP16 向量化
- [x] elu_fp16x8_packed — FP16 向量化与打包读写（原文称经 `half2` 约提升 2 倍；实际实现与条件见下方说明）
- [x] PyTorch 算子绑定与误差检查

## 读代码前先确认

ELU 在正半轴保持输入，非正半轴计算 $e^x-1$，本实现的 `alpha` 固定为 1。FP16 使用半精度指数和算术；`half2` 版本先计算两个分量的指数部分，再分别选择正、负分支。它没有可由 Python 传入的 `alpha` 参数。

这些实现每个线程独立处理元素，未使用共享内存或块内同步。`x4`、`x8` 描述每线程元素数，不代表相同倍数的性能提升。功能清单的 `[x]` 只表示原作者已实现该入口。

| 入口后缀 | 访问方式 | 不额外填充时的保守入门尺寸 |
| --- | --- | --- |
| 无后缀 | 每线程一个元素，先判断边界 | `0 < N <= INT_MAX-255`，保证原网格加法不溢出 |
| `_fp32x4` | `float4` 预读发生在判断前 | `N` 是 1024 的倍数，首地址 16 字节对齐 |
| `_fp16x2` | 每线程两个半精度元素 | `N` 是 512 的倍数，首地址 4 字节对齐 |
| `_fp16x8` | 四次 `half2` 预读发生在判断前 | `N` 是 2048 的倍数，首地址 4 字节对齐 |
| `_fp16x8_packed` | 先保护完整组，再 128 位读写 | `N` 是 8 的正整数倍，首地址 16 字节对齐 |

这里的 `N` 是输入总元素数，`INT_MAX` 是源码使用的有符号 `int` 的最大值。其他入口也应确保所有索引及网格计算不溢出。上表还假定输入输出同 CUDA 设备、同 dtype、连续、尺寸匹配、输出容量足够且不存在有害重叠；当前 CUDA 设备也应一致。包装没有完整验证这些条件。尺寸约定是阅读源码得到的保守条件，不是 GPU 测试结果。

具体区别：x2 完全没有核内边界判断；x4/x8 在保护分支之前已经读取。打包 x8 的核内尾部保护较完整，但主机网格先计算 `N/8` 再凑线程块，可能漏掉未成组尾部。因此不要把“存在 if”理解为支持任意长度，也不要把小输入直接套进这些入口。

## 测试与性能解释

先按[入门教程](../../docs/zh-CN/入门教程.md)检查环境，再在本目录运行下方原始命令。脚本依赖 PyTorch、NumPy、FlashInfer 和本机 CUDA 扩展构建环境；导入 `test.py` 就会编译扩展。原脚本使用大尺寸输入，不是轻量冒烟命令。

计时统计实际调用 `np.median(times)`，但原日志仍写 `mean time`。原地/非原地运算、输出是否预分配、预热、L2 状态和 `--use_fast_math` 都影响比较。下方保留原作者的历史数据，不能视为本次性能复测或速度保证。随机输入误差检查也不等于覆盖 NaN、Inf、极端值及所有形状。

本次只做翻译与静态一致性检查，未执行 CUDA 编译、GPU 运行、Compute Sanitizer 或性能测试。

ELU 原脚本的 FP16 基准会对 `a` 反复执行原地 ELU，后续比较针对变换后的输入；本次保留这一流程，只标注其含义。

## 运行测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 历史输出示例（非本次实测）

```bash
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.028800 ms
elu                            mean time: 0.035456 ms, speedup: 0.81
elu_fp32x4                     mean time: 0.025216 ms, speedup: 1.14
torch                          mean time: 0.014704 ms
elu_half                       mean time: 0.028656 ms, speedup: 0.51
elu_fp16x2                     mean time: 0.019488 ms, speedup: 0.75
elu_fp16x8                     mean time: 0.016880 ms, speedup: 0.87
elu_fp16x8_packed              mean time: 0.013120 ms, speedup: 1.12
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.049184 ms
elu                            mean time: 0.060048 ms, speedup: 0.82
elu_fp32x4                     mean time: 0.051952 ms, speedup: 0.95
torch                          mean time: 0.023264 ms
elu_half                       mean time: 0.050608 ms, speedup: 0.46
elu_fp16x2                     mean time: 0.033680 ms, speedup: 0.69
elu_fp16x8                     mean time: 0.031120 ms, speedup: 0.75
elu_fp16x8_packed              mean time: 0.023584 ms, speedup: 0.99
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.101216 ms
elu                            mean time: 0.114704 ms, speedup: 0.88
elu_fp32x4                     mean time: 0.135376 ms, speedup: 0.75
torch                          mean time: 0.060976 ms
elu_half                       mean time: 0.091312 ms, speedup: 0.67
elu_fp16x2                     mean time: 0.062320 ms, speedup: 0.98
elu_fp16x8                     mean time: 0.060640 ms, speedup: 1.01
elu_fp16x8_packed              mean time: 0.053696 ms, speedup: 1.14
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.049936 ms
elu                            mean time: 0.060656 ms, speedup: 0.82
elu_fp32x4                     mean time: 0.051344 ms, speedup: 0.97
torch                          mean time: 0.023744 ms
elu_half                       mean time: 0.054720 ms, speedup: 0.43
elu_fp16x2                     mean time: 0.037632 ms, speedup: 0.63
elu_fp16x8                     mean time: 0.034416 ms, speedup: 0.69
elu_fp16x8_packed              mean time: 0.023360 ms, speedup: 1.02
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.102304 ms
elu                            mean time: 0.132048 ms, speedup: 0.77
elu_fp32x4                     mean time: 0.106192 ms, speedup: 0.96
torch                          mean time: 0.047200 ms
elu_half                       mean time: 0.095584 ms, speedup: 0.49
elu_fp16x2                     mean time: 0.062480 ms, speedup: 0.76
elu_fp16x8                     mean time: 0.059296 ms, speedup: 0.80
elu_fp16x8_packed              mean time: 0.052192 ms, speedup: 0.90
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.204416 ms
elu                            mean time: 0.228704 ms, speedup: 0.89
elu_fp32x4                     mean time: 0.204416 ms, speedup: 1.00
torch                          mean time: 0.102416 ms
elu_half                       mean time: 0.171552 ms, speedup: 0.60
elu_fp16x2                     mean time: 0.116032 ms, speedup: 0.88
elu_fp16x8                     mean time: 0.109552 ms, speedup: 0.93
elu_fp16x8_packed              mean time: 0.103024 ms, speedup: 0.99
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.101312 ms
elu                            mean time: 0.116368 ms, speedup: 0.87
elu_fp32x4                     mean time: 0.102880 ms, speedup: 0.98
torch                          mean time: 0.046768 ms
elu_half                       mean time: 0.093760 ms, speedup: 0.50
elu_fp16x2                     mean time: 0.062336 ms, speedup: 0.75
elu_fp16x8                     mean time: 0.060640 ms, speedup: 0.77
elu_fp16x8_packed              mean time: 0.052688 ms, speedup: 0.89
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.205088 ms
elu                            mean time: 0.233424 ms, speedup: 0.88
elu_fp32x4                     mean time: 0.207152 ms, speedup: 0.99
torch                          mean time: 0.102016 ms
elu_half                       mean time: 0.183840 ms, speedup: 0.55
elu_fp16x2                     mean time: 0.121744 ms, speedup: 0.84
elu_fp16x8                     mean time: 0.111776 ms, speedup: 0.91
elu_fp16x8_packed              mean time: 0.104096 ms, speedup: 0.98
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.403728 ms
elu                            mean time: 0.438784 ms, speedup: 0.92
elu_fp32x4                     mean time: 0.426912 ms, speedup: 0.95
torch                          mean time: 0.203392 ms
elu_half                       mean time: 0.325744 ms, speedup: 0.62
elu_fp16x2                     mean time: 0.227152 ms, speedup: 0.90
elu_fp16x8                     mean time: 0.207440 ms, speedup: 0.98
elu_fp16x8_packed              mean time: 0.204368 ms, speedup: 1.00
```
