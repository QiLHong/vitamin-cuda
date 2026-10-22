# hardswish

[中文首页](../../README.md) · [术语表](../../docs/zh-CN/术语表.md) · [英文原文](./readme.en.md)

## 功能概览

Hardswish 激活函数核函数。

- [x] hardswish — FP32 / FP16
- [x] hardswish_fp32x4 — FP32 四元素向量化（与源码入口对应）
- [x] hardswish_fp16x2 — FP16 向量化
- [x] hardswish_fp16x8 — FP16 向量化
- [x] hardswish_fp16x8_packed — FP16 向量化与打包读写
- [x] PyTorch 算子绑定与误差检查

## 读代码前先确认

对有限实数计算 $x\min(\max(x+3,0),6)/6$。`half`、`half2` 保持半精度计算，乘数 `1/6` 会舍入；这与先用 FP32 计算再转换不保证相同的逐位结果。

这些实现每个线程独立处理元素，未使用共享内存或块内同步。`x4`、`x8` 描述每线程元素数，不代表相同倍数的性能提升。功能清单的 `[x]` 只表示原作者已实现该入口。

| 入口后缀 | 访问方式 | 不额外填充时的保守入门尺寸 |
| --- | --- | --- |
| 无后缀 | 每线程一个元素，先判断边界 | `0 < N <= INT_MAX-255`，保证原网格加法不溢出 |
| `_fp32x4` | `float4` 预读发生在判断前 | `N` 是 1024 的倍数，首地址 16 字节对齐 |
| `_fp16x2` | 每线程两个半精度元素 | `N` 是 512 的倍数，首地址 4 字节对齐 |
| `_fp16x8` | 四次 `half2` 预读发生在判断前 | `N` 是 2048 的倍数，首地址 4 字节对齐 |
| `_fp16x8_packed` | 先保护完整组，再 128 位读写 | `N` 是 8 的正整数倍，首地址 16 字节对齐 |

这里的 `N` 是输入总元素数，`INT_MAX` 是源码使用的有符号 `int` 的最大值。其他入口也应确保所有索引及网格计算不溢出。上表还假定输入输出同 CUDA 设备、同 dtype、连续、尺寸匹配、输出容量足够且不存在有害重叠；当前 CUDA 设备也应一致。包装没有完整验证这些条件。尺寸约定是阅读源码得到的保守条件，不是 GPU 测试结果。

具体区别：x2 的单元素 else 分支没有检查 `idx<N`，多余线程可能越界；x4/x8 在保护分支之前已经读取。打包 x8 的核内尾部保护较完整，但主机网格先计算 `N/8` 再凑线程块，可能漏掉未成组尾部。因此不要把“存在 if”理解为支持任意长度，也不要把小输入直接套进这些入口。

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
torch                          mean time: 0.027616 ms
hardswish                      mean time: 0.036224 ms, speedup: 0.76
hardswish_fp32x4               mean time: 0.025792 ms, speedup: 1.07
torch                          mean time: 0.014672 ms
hardswish_half                 mean time: 0.028896 ms, speedup: 0.51
hardswish_fp16x2               mean time: 0.020880 ms, speedup: 0.70
hardswish_fp16x8               mean time: 0.013408 ms, speedup: 1.09
hardswish_fp16x8_packed        mean time: 0.013200 ms, speedup: 1.11
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.049104 ms
hardswish                      mean time: 0.060176 ms, speedup: 0.82
hardswish_fp32x4               mean time: 0.052752 ms, speedup: 0.93
torch                          mean time: 0.023696 ms
hardswish_half                 mean time: 0.049360 ms, speedup: 0.48
hardswish_fp16x2               mean time: 0.033952 ms, speedup: 0.70
hardswish_fp16x8               mean time: 0.027312 ms, speedup: 0.87
hardswish_fp16x8_packed        mean time: 0.023488 ms, speedup: 1.01
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.101328 ms
hardswish                      mean time: 0.115744 ms, speedup: 0.88
hardswish_fp32x4               mean time: 0.103520 ms, speedup: 0.98
torch                          mean time: 0.046336 ms
hardswish_half                 mean time: 0.087264 ms, speedup: 0.53
hardswish_fp16x2               mean time: 0.061280 ms, speedup: 0.76
hardswish_fp16x8               mean time: 0.055824 ms, speedup: 0.83
hardswish_fp16x8_packed        mean time: 0.052560 ms, speedup: 0.88
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.048832 ms
hardswish                      mean time: 0.060064 ms, speedup: 0.81
hardswish_fp32x4               mean time: 0.050528 ms, speedup: 0.97
torch                          mean time: 0.023024 ms
hardswish_half                 mean time: 0.048384 ms, speedup: 0.48
hardswish_fp16x2               mean time: 0.033664 ms, speedup: 0.68
hardswish_fp16x8               mean time: 0.027424 ms, speedup: 0.84
hardswish_fp16x8_packed        mean time: 0.023600 ms, speedup: 0.98
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.100768 ms
hardswish                      mean time: 0.114112 ms, speedup: 0.88
hardswish_fp32x4               mean time: 0.105440 ms, speedup: 0.96
torch                          mean time: 0.046112 ms
hardswish_half                 mean time: 0.086736 ms, speedup: 0.53
hardswish_fp16x2               mean time: 0.061040 ms, speedup: 0.76
hardswish_fp16x8               mean time: 0.053360 ms, speedup: 0.86
hardswish_fp16x8_packed        mean time: 0.052096 ms, speedup: 0.89
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.203008 ms
hardswish                      mean time: 0.225024 ms, speedup: 0.90
hardswish_fp32x4               mean time: 0.203728 ms, speedup: 1.00
torch                          mean time: 0.101424 ms
hardswish_half                 mean time: 0.162784 ms, speedup: 0.62
hardswish_fp16x2               mean time: 0.115248 ms, speedup: 0.88
hardswish_fp16x8               mean time: 0.103520 ms, speedup: 0.98
hardswish_fp16x8_packed        mean time: 0.102352 ms, speedup: 0.99
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.100896 ms
hardswish                      mean time: 0.115520 ms, speedup: 0.87
hardswish_fp32x4               mean time: 0.102848 ms, speedup: 0.98
torch                          mean time: 0.046064 ms
hardswish_half                 mean time: 0.087520 ms, speedup: 0.53
hardswish_fp16x2               mean time: 0.061792 ms, speedup: 0.75
hardswish_fp16x8               mean time: 0.054832 ms, speedup: 0.84
hardswish_fp16x8_packed        mean time: 0.068608 ms, speedup: 0.67
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.203840 ms
hardswish                      mean time: 0.227024 ms, speedup: 0.90
hardswish_fp32x4               mean time: 0.205344 ms, speedup: 0.99
torch                          mean time: 0.101456 ms
hardswish_half                 mean time: 0.192768 ms, speedup: 0.53
hardswish_fp16x2               mean time: 0.145088 ms, speedup: 0.70
hardswish_fp16x8               mean time: 0.136368 ms, speedup: 0.74
hardswish_fp16x8_packed        mean time: 0.135616 ms, speedup: 0.75
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.516928 ms
hardswish                      mean time: 0.423056 ms, speedup: 1.22
hardswish_fp32x4               mean time: 0.401536 ms, speedup: 1.29
torch                          mean time: 0.202944 ms
hardswish_half                 mean time: 0.314096 ms, speedup: 0.65
hardswish_fp16x2               mean time: 0.227440 ms, speedup: 0.89
hardswish_fp16x8               mean time: 0.206576 ms, speedup: 0.98
hardswish_fp16x8_packed        mean time: 0.205008 ms, speedup: 0.99
```
