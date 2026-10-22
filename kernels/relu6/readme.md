# relu6

[中文首页](../../README.md) · [术语表](../../docs/zh-CN/术语表.md) · [英文原文](./readme.en.md)

## 功能概览

ReLU6 激活函数核函数。

- [x] relu6 — FP32 / FP16
- [x] relu6_fp32x4 — FP32 四元素向量化（与源码入口对应）
- [x] relu6_fp16x2 — FP16 向量化
- [x] relu6_fp16x8 — FP16 向量化
- [x] relu6_fp16x8_packed — FP16 向量化与打包读写
- [x] PyTorch 算子绑定与误差检查

## 读代码前先确认

对有限实数计算 $\min(\max(x,0),6)$。三个设备函数重载分别处理 `float`、`half`、`half2`；`half2` 的两个分量独立截断，并不互相规约。NaN、无穷大与有符号零的行为应按所用 intrinsic 另做验证。

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

## 运行测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 历史输出示例（非本次实测）

```bash
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.029376 ms
relu6                          mean time: 0.037056 ms, speedup: 0.79
relu6_fp32x4                   mean time: 0.026176 ms, speedup: 1.12
torch                          mean time: 0.013648 ms
relu6_half                     mean time: 0.027648 ms, speedup: 0.49
relu6_fp16x2                   mean time: 0.020512 ms, speedup: 0.67
relu6_fp16x8                   mean time: 0.013424 ms, speedup: 1.02
relu6_fp16x8_packed            mean time: 0.013152 ms, speedup: 1.04
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.048800 ms
relu6                          mean time: 0.060016 ms, speedup: 0.81
relu6_fp32x4                   mean time: 0.052432 ms, speedup: 0.93
torch                          mean time: 0.023440 ms
relu6_half                     mean time: 0.048608 ms, speedup: 0.48
relu6_fp16x2                   mean time: 0.033744 ms, speedup: 0.69
relu6_fp16x8                   mean time: 0.023696 ms, speedup: 0.99
relu6_fp16x8_packed            mean time: 0.024240 ms, speedup: 0.97
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.101152 ms
relu6                          mean time: 0.114240 ms, speedup: 0.89
relu6_fp32x4                   mean time: 0.103472 ms, speedup: 0.98
torch                          mean time: 0.046288 ms
relu6_half                     mean time: 0.086912 ms, speedup: 0.53
relu6_fp16x2                   mean time: 0.060560 ms, speedup: 0.76
relu6_fp16x8                   mean time: 0.053456 ms, speedup: 0.87
relu6_fp16x8_packed            mean time: 0.053696 ms, speedup: 0.86
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.049040 ms
relu6                          mean time: 0.060208 ms, speedup: 0.81
relu6_fp32x4                   mean time: 0.050560 ms, speedup: 0.97
torch                          mean time: 0.021792 ms
relu6_half                     mean time: 0.048416 ms, speedup: 0.45
relu6_fp16x2                   mean time: 0.033392 ms, speedup: 0.65
relu6_fp16x8                   mean time: 0.024480 ms, speedup: 0.89
relu6_fp16x8_packed            mean time: 0.023392 ms, speedup: 0.93
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.100592 ms
relu6                          mean time: 0.114544 ms, speedup: 0.88
relu6_fp32x4                   mean time: 0.105776 ms, speedup: 0.95
torch                          mean time: 0.046288 ms
relu6_half                     mean time: 0.086832 ms, speedup: 0.53
relu6_fp16x2                   mean time: 0.060544 ms, speedup: 0.76
relu6_fp16x8                   mean time: 0.052048 ms, speedup: 0.89
relu6_fp16x8_packed            mean time: 0.052272 ms, speedup: 0.89
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.202784 ms
relu6                          mean time: 0.225776 ms, speedup: 0.90
relu6_fp32x4                   mean time: 0.204032 ms, speedup: 0.99
torch                          mean time: 0.100800 ms
relu6_half                     mean time: 0.160720 ms, speedup: 0.63
relu6_fp16x2                   mean time: 0.113088 ms, speedup: 0.89
relu6_fp16x8                   mean time: 0.103248 ms, speedup: 0.98
relu6_fp16x8_packed            mean time: 0.102864 ms, speedup: 0.98
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.100272 ms
relu6                          mean time: 0.114928 ms, speedup: 0.87
relu6_fp32x4                   mean time: 0.103392 ms, speedup: 0.97
torch                          mean time: 0.046000 ms
relu6_half                     mean time: 0.088832 ms, speedup: 0.52
relu6_fp16x2                   mean time: 0.060416 ms, speedup: 0.76
relu6_fp16x8                   mean time: 0.052560 ms, speedup: 0.88
relu6_fp16x8_packed            mean time: 0.052544 ms, speedup: 0.88
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.202960 ms
relu6                          mean time: 0.225776 ms, speedup: 0.90
relu6_fp32x4                   mean time: 0.205088 ms, speedup: 0.99
torch                          mean time: 0.102128 ms
relu6_half                     mean time: 0.162352 ms, speedup: 0.63
relu6_fp16x2                   mean time: 0.113600 ms, speedup: 0.90
relu6_fp16x8                   mean time: 0.103840 ms, speedup: 0.98
relu6_fp16x8_packed            mean time: 0.103616 ms, speedup: 0.99
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.401920 ms
relu6                          mean time: 0.414784 ms, speedup: 0.97
relu6_fp32x4                   mean time: 0.400976 ms, speedup: 1.00
torch                          mean time: 0.258784 ms
relu6_half                     mean time: 0.356512 ms, speedup: 0.73
relu6_fp16x2                   mean time: 0.268176 ms, speedup: 0.96
relu6_fp16x8                   mean time: 0.204752 ms, speedup: 1.26
relu6_fp16x8_packed            mean time: 0.204704 ms, speedup: 1.26
```
