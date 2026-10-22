# GELU

[中文首页](../../README.md) · [术语表](../../docs/zh-CN/术语表.md) · [英文原文](./readme.en.md)

## 功能概览

GELU 激活函数核函数。

- [x] gelu — FP32 / FP16
- [x] gelu_fp32x4 — FP32 四元素向量化（与源码入口对应）
- [x] gelu_fp16x2 — FP16 向量化
- [x] gelu_fp16x8 — FP16 向量化
- [x] gelu_fp16x8_packed — FP16 向量化与打包读写（原文称经 `half2` 约提升 2 倍；实际实现与条件见下方说明）
- [x] PyTorch 算子绑定与误差检查

## 读代码前先确认

本实现使用 $0.5x(1+\operatorname{erf}(x/\sqrt{2}))$。FP16 标量先转 `float` 计算 `erff` 再转回；`half2` 先解包为 `float2`，两个分量分别计算。打包 x8 版本实际循环处理八个 `half`，不是四次 `half2` 算术，因此不能仅凭 x8 名称推断计算指令或加速倍数。

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

原文说明：这里的 PyTorch GELU 没有使用预分配输出的快速路径，因此微基准可能偏向自定义核函数。在紧密循环中，分配器噪声减弱，PyTorch 也可能表现得很快；公平比较时应统一分配与计时条件。

### 历史输出示例（非本次实测）

```bash
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.025856 ms
gelu                           mean time: 0.058672 ms, speedup: 0.44
gelu_fp32x4                    mean time: 0.058112 ms, speedup: 0.44
torch                          mean time: 0.013280 ms
gelu_half                      mean time: 0.029008 ms, speedup: 0.46
gelu_fp16x2                    mean time: 0.019680 ms, speedup: 0.67
gelu_fp16x8                    mean time: 0.017184 ms, speedup: 0.77
gelu_fp16x8_packed             mean time: 0.013152 ms, speedup: 1.01
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.051248 ms
gelu                           mean time: 0.108128 ms, speedup: 0.47
gelu_fp32x4                    mean time: 0.108208 ms, speedup: 0.47
torch                          mean time: 0.023712 ms
gelu_half                      mean time: 0.051632 ms, speedup: 0.46
gelu_fp16x2                    mean time: 0.033888 ms, speedup: 0.70
gelu_fp16x8                    mean time: 0.031392 ms, speedup: 0.76
gelu_fp16x8_packed             mean time: 0.023232 ms, speedup: 1.02
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.101936 ms
gelu                           mean time: 0.213152 ms, speedup: 0.48
gelu_fp32x4                    mean time: 0.211312 ms, speedup: 0.48
torch                          mean time: 0.051680 ms
gelu_half                      mean time: 0.091008 ms, speedup: 0.57
gelu_fp16x2                    mean time: 0.062496 ms, speedup: 0.83
gelu_fp16x8                    mean time: 0.059264 ms, speedup: 0.87
gelu_fp16x8_packed             mean time: 0.052112 ms, speedup: 0.99
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.051920 ms
gelu                           mean time: 0.108144 ms, speedup: 0.48
gelu_fp32x4                    mean time: 0.108768 ms, speedup: 0.48
torch                          mean time: 0.023680 ms
gelu_half                      mean time: 0.052352 ms, speedup: 0.45
gelu_fp16x2                    mean time: 0.035712 ms, speedup: 0.66
gelu_fp16x8                    mean time: 0.032384 ms, speedup: 0.73
gelu_fp16x8_packed             mean time: 0.025120 ms, speedup: 0.94
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.132928 ms
gelu                           mean time: 0.219360 ms, speedup: 0.61
gelu_fp32x4                    mean time: 0.218928 ms, speedup: 0.61
torch                          mean time: 0.052048 ms
gelu_half                      mean time: 0.097104 ms, speedup: 0.54
gelu_fp16x2                    mean time: 0.064736 ms, speedup: 0.80
gelu_fp16x8                    mean time: 0.059072 ms, speedup: 0.88
gelu_fp16x8_packed             mean time: 0.051632 ms, speedup: 1.01
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.206048 ms
gelu                           mean time: 0.450112 ms, speedup: 0.46
gelu_fp32x4                    mean time: 0.423776 ms, speedup: 0.49
torch                          mean time: 0.102672 ms
gelu_half                      mean time: 0.177920 ms, speedup: 0.58
gelu_fp16x2                    mean time: 0.124752 ms, speedup: 0.82
gelu_fp16x8                    mean time: 0.110992 ms, speedup: 0.93
gelu_fp16x8_packed             mean time: 0.103376 ms, speedup: 0.99
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.101088 ms
gelu                           mean time: 0.226736 ms, speedup: 0.45
gelu_fp32x4                    mean time: 0.240160 ms, speedup: 0.42
torch                          mean time: 0.050608 ms
gelu_half                      mean time: 0.098880 ms, speedup: 0.51
gelu_fp16x2                    mean time: 0.065040 ms, speedup: 0.78
gelu_fp16x8                    mean time: 0.060112 ms, speedup: 0.84
gelu_fp16x8_packed             mean time: 0.051856 ms, speedup: 0.98
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.205088 ms
gelu                           mean time: 0.458576 ms, speedup: 0.45
gelu_fp32x4                    mean time: 0.448736 ms, speedup: 0.46
torch                          mean time: 0.103840 ms
gelu_half                      mean time: 0.176976 ms, speedup: 0.59
gelu_fp16x2                    mean time: 0.122416 ms, speedup: 0.85
gelu_fp16x8                    mean time: 0.107280 ms, speedup: 0.97
gelu_fp16x8_packed             mean time: 0.102960 ms, speedup: 1.01
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.406112 ms
gelu                           mean time: 0.831136 ms, speedup: 0.49
gelu_fp32x4                    mean time: 0.822448 ms, speedup: 0.49
torch                          mean time: 0.205120 ms
gelu_half                      mean time: 0.352656 ms, speedup: 0.58
gelu_fp16x2                    mean time: 0.231968 ms, speedup: 0.88
gelu_fp16x8                    mean time: 0.209296 ms, speedup: 0.98
gelu_fp16x8_packed             mean time: 0.204832 ms, speedup: 1.00
```
