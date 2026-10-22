# 点积（Dot Product）

[英文原文](./readme.en.md) · [中文入门教程](../../docs/zh-CN/入门教程.md) · [术语表](../../docs/zh-CN/术语表.md)

> 本页保留原功能清单和历史输出；功能勾选不表示本次已运行验证。

## 概述

点积核函数，包含标量版本和向量化版本。

- [x] dot_product — FP32 / FP16
- [x] dot_product_fp32x4 — FP32 向量化
- [x] dot_product_fp16x2 — FP16 向量化
- [x] dot_product_fp16x8 — FP16 向量化、打包读写
- [x] PyTorch 算子绑定与结果差异检查

## 中文导读与适用范围

点积计算 `sum(a[i] * b[i])`，返回只有一个元素的张量。线程先沿网格步长累积局部乘积，再通过线程束交换、共享内存和第一线程束完成块内规约；各块用 `atomicAdd` 加到预先清零的 FP32 输出，最后转换回输入 dtype。

标量入口支持 FP32/FP16。半精度标量路径的表达式先做 `a[i] * b[i]`，再转为 FP32；`half2` 路径也先用 `__hmul2` 做半精度乘法，再转 FP32 累加。因此“FP32 累加”不等同于“乘法和所有中间结果都为 FP32”。浮点加法的分组及原子累加顺序还可能影响舍入结果。

FP32x4、FP16x2、FP16x8 版本的总元素数应分别为 4、2、8 的正整数倍；循环只检查组起点，不能保护不完整的最后一组。`float4`/128 位打包访问要求相应的 16 字节对齐，`half2` 要求 4 字节对齐。调用方还需保证两输入元素数、dtype 与设备一致；包装没有检查所有关系。

原概览中的 `dot_product_fp16x8` 对应当前 Python 实际入口 `dot_product_fp16x8_packed`。测试脚本用绝对误差或相对误差检查结果，参考值为零时相对误差表达式可能出现除零；原测试与容差保持不变。

## 运行测试

需要可用的 NVIDIA GPU、匹配的 CUDA/PyTorch 编译环境，以及脚本导入的 NumPy、FlashInfer 等依赖。在当前模块目录执行下列命令；本次未进行 CUDA 编译、GPU 正确性或性能测试。

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 历史示例输出

以下为原仓库保留的记录（含原英文日志和数值），不是本次翻译产生的运行结果。Python 计时统计实际使用中位数 `np.median`，原日志标签 `mean time` 保留不变。

```bash
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.037600 ms
dot_product                    mean time: 0.045568 ms, speedup: 0.83
dot_product_fp32x4             mean time: 0.053104 ms, speedup: 0.71
torch                          mean time: 0.025648 ms
dot_product                    mean time: 0.041696 ms, speedup: 0.62
dot_product_fp16x2             mean time: 0.028704 ms, speedup: 0.89
dot_product_fp16x8_packed      mean time: 0.029376 ms, speedup: 0.87
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.059488 ms
dot_product                    mean time: 0.071280 ms, speedup: 0.83
dot_product_fp32x4             mean time: 0.067024 ms, speedup: 0.89
torch                          mean time: 0.036784 ms
dot_product                    mean time: 0.064608 ms, speedup: 0.57
dot_product_fp16x2             mean time: 0.047360 ms, speedup: 0.78
dot_product_fp16x8_packed      mean time: 0.053776 ms, speedup: 0.68
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.108720 ms
dot_product                    mean time: 0.120128 ms, speedup: 0.91
dot_product_fp32x4             mean time: 0.128256 ms, speedup: 0.85
torch                          mean time: 0.061952 ms
dot_product                    mean time: 0.090496 ms, speedup: 0.68
dot_product_fp16x2             mean time: 0.082512 ms, speedup: 0.75
dot_product_fp16x8_packed      mean time: 0.073552 ms, speedup: 0.84
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.060176 ms
dot_product                    mean time: 0.077376 ms, speedup: 0.78
dot_product_fp32x4             mean time: 0.070384 ms, speedup: 0.85
torch                          mean time: 0.036768 ms
dot_product                    mean time: 0.063952 ms, speedup: 0.57
dot_product_fp16x2             mean time: 0.047280 ms, speedup: 0.78
dot_product_fp16x8_packed      mean time: 0.052144 ms, speedup: 0.71
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.108384 ms
dot_product                    mean time: 0.121664 ms, speedup: 0.89
dot_product_fp32x4             mean time: 0.128464 ms, speedup: 0.84
torch                          mean time: 0.061856 ms
dot_product                    mean time: 0.090528 ms, speedup: 0.68
dot_product_fp16x2             mean time: 0.080992 ms, speedup: 0.76
dot_product_fp16x8_packed      mean time: 0.072512 ms, speedup: 0.85
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.209552 ms
dot_product                    mean time: 0.223408 ms, speedup: 0.94
dot_product_fp32x4             mean time: 0.223312 ms, speedup: 0.94
torch                          mean time: 0.113056 ms
dot_product                    mean time: 0.181024 ms, speedup: 0.62
dot_product_fp16x2             mean time: 0.124512 ms, speedup: 0.91
dot_product_fp16x8_packed      mean time: 0.123984 ms, speedup: 0.91
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.107712 ms
dot_product                    mean time: 0.123280 ms, speedup: 0.87
dot_product_fp32x4             mean time: 0.115728 ms, speedup: 0.93
torch                          mean time: 0.061936 ms
dot_product                    mean time: 0.090608 ms, speedup: 0.68
dot_product_fp16x2             mean time: 0.080512 ms, speedup: 0.77
dot_product_fp16x8_packed      mean time: 0.071904 ms, speedup: 0.86
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.209472 ms
dot_product                    mean time: 0.226800 ms, speedup: 0.92
dot_product_fp32x4             mean time: 0.215968 ms, speedup: 0.97
torch                          mean time: 0.113264 ms
dot_product                    mean time: 0.174976 ms, speedup: 0.65
dot_product_fp16x2             mean time: 0.128368 ms, speedup: 0.88
dot_product_fp16x8_packed      mean time: 0.128128 ms, speedup: 0.88
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.403200 ms
dot_product                    mean time: 0.418528 ms, speedup: 0.96
dot_product_fp32x4             mean time: 0.407856 ms, speedup: 0.99
torch                          mean time: 0.214192 ms
dot_product                    mean time: 0.279984 ms, speedup: 0.77
dot_product_fp16x2             mean time: 0.235008 ms, speedup: 0.91
dot_product_fp16x8_packed      mean time: 0.220992 ms, speedup: 0.97
```
