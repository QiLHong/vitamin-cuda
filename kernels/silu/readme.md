# silu

[中文首页](../../README.md) · [术语表](../../docs/zh-CN/术语表.md) · [英文原文](./readme.en.md)

## 功能概览

SiLU（Swish）激活函数核函数。

- [x] silu — FP32 / FP16
- [x] silu_fp32x4 — FP32 四元素向量化（与源码入口对应）
- [x] silu_fp16x2 — FP16 向量化
- [x] silu_fp16x8 — FP16 向量化
- [x] silu_fp16x8_packed — FP16 向量化与打包读写
- [x] PyTorch 算子绑定与误差检查

## 读代码前先确认

计算 $x/(1+e^{-x})$。FP32 使用 `expf` 与 `__frcp_rn`；FP16/`half2` 使用对应半精度指数、倒数和乘法。极端输入下，指数溢出、下溢和逐步舍入可能影响结果，不能仅凭数学公式推断全部浮点边界行为。

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
torch                          mean time: 0.028128 ms
silu                           mean time: 0.037760 ms, speedup: 0.74
silu_fp32x4                    mean time: 0.023744 ms, speedup: 1.18
torch                          mean time: 0.017568 ms
silu_half                      mean time: 0.028656 ms, speedup: 0.61
silu_fp16x2                    mean time: 0.020832 ms, speedup: 0.84
silu_fp16x8                    mean time: 0.016560 ms, speedup: 1.06
silu_fp16x8_packed             mean time: 0.013456 ms, speedup: 1.31
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.052512 ms
silu                           mean time: 0.061664 ms, speedup: 0.85
silu_fp32x4                    mean time: 0.052544 ms, speedup: 1.00
torch                          mean time: 0.025472 ms
silu_half                      mean time: 0.050416 ms, speedup: 0.51
silu_fp16x2                    mean time: 0.033920 ms, speedup: 0.75
silu_fp16x8                    mean time: 0.031248 ms, speedup: 0.82
silu_fp16x8_packed             mean time: 0.023440 ms, speedup: 1.09
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.100160 ms
silu                           mean time: 0.117312 ms, speedup: 0.85
silu_fp32x4                    mean time: 0.104800 ms, speedup: 0.96
torch                          mean time: 0.049792 ms
silu_half                      mean time: 0.089088 ms, speedup: 0.56
silu_fp16x2                    mean time: 0.061552 ms, speedup: 0.81
silu_fp16x8                    mean time: 0.057024 ms, speedup: 0.87
silu_fp16x8_packed             mean time: 0.050928 ms, speedup: 0.98
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.048288 ms
silu                           mean time: 0.061952 ms, speedup: 0.78
silu_fp32x4                    mean time: 0.052320 ms, speedup: 0.92
torch                          mean time: 0.025776 ms
silu_half                      mean time: 0.050128 ms, speedup: 0.51
silu_fp16x2                    mean time: 0.034176 ms, speedup: 0.75
silu_fp16x8                    mean time: 0.031312 ms, speedup: 0.82
silu_fp16x8_packed             mean time: 0.023552 ms, speedup: 1.09
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.099248 ms
silu                           mean time: 0.118000 ms, speedup: 0.84
silu_fp32x4                    mean time: 0.104224 ms, speedup: 0.95
torch                          mean time: 0.049648 ms
silu_half                      mean time: 0.092912 ms, speedup: 0.53
silu_fp16x2                    mean time: 0.062720 ms, speedup: 0.79
silu_fp16x8                    mean time: 0.060096 ms, speedup: 0.83
silu_fp16x8_packed             mean time: 0.054016 ms, speedup: 0.92
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.203888 ms
silu                           mean time: 0.232784 ms, speedup: 0.88
silu_fp32x4                    mean time: 0.208960 ms, speedup: 0.98
torch                          mean time: 0.099952 ms
silu_half                      mean time: 0.171680 ms, speedup: 0.58
silu_fp16x2                    mean time: 0.118368 ms, speedup: 0.84
silu_fp16x8                    mean time: 0.109200 ms, speedup: 0.92
silu_fp16x8_packed             mean time: 0.105152 ms, speedup: 0.95
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.100416 ms
silu                           mean time: 0.117888 ms, speedup: 0.85
silu_fp32x4                    mean time: 0.106016 ms, speedup: 0.95
torch                          mean time: 0.049984 ms
silu_half                      mean time: 0.089440 ms, speedup: 0.56
silu_fp16x2                    mean time: 0.061104 ms, speedup: 0.82
silu_fp16x8                    mean time: 0.058224 ms, speedup: 0.86
silu_fp16x8_packed             mean time: 0.051808 ms, speedup: 0.96
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.202368 ms
silu                           mean time: 0.231408 ms, speedup: 0.87
silu_fp32x4                    mean time: 0.208544 ms, speedup: 0.97
torch                          mean time: 0.099840 ms
silu_half                      mean time: 0.177616 ms, speedup: 0.56
silu_fp16x2                    mean time: 0.120464 ms, speedup: 0.83
silu_fp16x8                    mean time: 0.109776 ms, speedup: 0.91
silu_fp16x8_packed             mean time: 0.105840 ms, speedup: 0.94
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.404112 ms
silu                           mean time: 0.431648 ms, speedup: 0.94
silu_fp32x4                    mean time: 0.404432 ms, speedup: 1.00
torch                          mean time: 0.202048 ms
silu_half                      mean time: 0.321248 ms, speedup: 0.63
silu_fp16x2                    mean time: 0.229792 ms, speedup: 0.88
silu_fp16x8                    mean time: 0.211008 ms, speedup: 0.96
silu_fp16x8_packed             mean time: 0.206448 ms, speedup: 0.98
```
