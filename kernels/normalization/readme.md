# norm 1d：一维归一化

[英文原文](./readme.en.md) · [中文入门教程](../../docs/zh-CN/入门教程.md) · [术语表](../../docs/zh-CN/术语表.md)

> 本页保留原功能清单和历史输出；功能勾选不表示本次已运行验证。

## 概述

归一化核函数计算 y = (x - mean) / std，其中 mean 为行均值，std 为行标准差。

- [x] 朴素 Torch 归一化基准
- [x] norm — FP32
- [x] norm — FP32x4
- [x] norm — FP16
- [x] norm — FP16x8，打包读写
- [x] norm — FP32x4，按行内分块拆分计算（split-k）
- [x] PyTorch 算子绑定与结果差异检查

## 中文导读与适用范围

输入输出按连续二维张量 `[bs, n]` 解释，每一行独立计算 `mean = sum(x) / n`、`std = sqrt(sum(x*x) / n - mean*mean)`，再写出 `(x-mean)/std`。这里使用总体方差，Python 对照设置 `unbiased=False`。这不是包含可学习缩放和偏置的完整 LayerNorm 接口。

普通版本一个线程块处理一行，先汇总一阶和二阶统计量，再读取输入完成归一化。FP16 输入转换为 FP32 进行统计和归一化计算，写出时舍入回 FP16。split-k 版本把一行拆给多个线程块：第一轮写部分和及平方和，第二轮合并整行统计并写出当前分块；两个核函数在同一 CUDA 流中顺序执行。

FP32/FP16 标量版本逐元素判断范围。FP32x4（包括 split-k）要求行长为 4 的正整数倍，FP16x8 打包版本要求行长为 8 的正整数倍，并保证 16 字节地址对齐；只检查组起点不能保护任意尾部。包装仅检查 CUDA 与连续性，未核对 dtype、二维形状、输出容量及设备一致性。

原实现没有在标准差中加入 epsilon，也没有把舍入误差造成的负方差截断为零。常量行可能除零，`E[x²]-E[x]²` 也可能有消减误差；文档与注释保留这些事实，没有改变公式。当前 `test.py` 默认只调用 `test()`，`test_split_k()` 调用仍被注释掉；下方 split-k 日志是原仓库历史记录。

## 运行测试

需要可用的 NVIDIA GPU、匹配的 CUDA/PyTorch 编译环境，以及脚本导入的 NumPy、FlashInfer 等依赖。在当前模块目录执行下列命令；本次未进行 CUDA 编译、GPU 正确性或性能测试。

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 历史示例输出

以下为原仓库保留的记录（含原英文日志和数值），不是本次翻译产生的运行结果。Python 计时统计实际使用中位数 `np.median`，原日志标签 `mean time` 保留不变。

#### split-k 版本测试

```bash
####################################################################################################
n: 1, m: 8192
torch                          mean time: 0.007360 ms
norm_fp32                      mean time: 0.009280 ms, speedup: 0.79
norm_fp32x4                    mean time: 0.008352 ms, speedup: 0.88
norm_fp32x4_split_k            mean time: 0.011408 ms, speedup: 0.65
####################################################################################################
n: 1, m: 12800
torch                          mean time: 0.021808 ms
norm_fp32                      mean time: 0.010592 ms, speedup: 2.06
norm_fp32x4                    mean time: 0.009136 ms, speedup: 2.39
norm_fp32x4_split_k            mean time: 0.011392 ms, speedup: 1.91
####################################################################################################
n: 1, m: 102400
torch                          mean time: 0.038160 ms
norm_fp32                      mean time: 0.077584 ms, speedup: 0.49
norm_fp32x4                    mean time: 0.031840 ms, speedup: 1.20
norm_fp32x4_split_k            mean time: 0.012736 ms, speedup: 3.00
####################################################################################################
n: 4, m: 8192
torch                          mean time: 0.011184 ms
norm_fp32                      mean time: 0.009280 ms, speedup: 1.21
norm_fp32x4                    mean time: 0.008896 ms, speedup: 1.26
norm_fp32x4_split_k            mean time: 0.011392 ms, speedup: 0.98
####################################################################################################
n: 4, m: 12800
torch                          mean time: 0.013664 ms
norm_fp32                      mean time: 0.011312 ms, speedup: 1.21
norm_fp32x4                    mean time: 0.009392 ms, speedup: 1.45
norm_fp32x4_split_k            mean time: 0.011632 ms, speedup: 1.17
####################################################################################################
n: 4, m: 102400
torch                          mean time: 0.072192 ms
norm_fp32                      mean time: 0.081824 ms, speedup: 0.88
norm_fp32x4                    mean time: 0.036272 ms, speedup: 1.99
norm_fp32x4_split_k            mean time: 0.018128 ms, speedup: 3.98
####################################################################################################
n: 8, m: 8192
torch                          mean time: 0.011248 ms
norm_fp32                      mean time: 0.009296 ms, speedup: 1.21
norm_fp32x4                    mean time: 0.009216 ms, speedup: 1.22
norm_fp32x4_split_k            mean time: 0.012080 ms, speedup: 0.93
####################################################################################################
n: 8, m: 12800
torch                          mean time: 0.013600 ms
norm_fp32                      mean time: 0.011888 ms, speedup: 1.14
norm_fp32x4                    mean time: 0.010832 ms, speedup: 1.26
norm_fp32x4_split_k            mean time: 0.012864 ms, speedup: 1.06
####################################################################################################
n: 8, m: 102400
torch                          mean time: 0.072480 ms
norm_fp32                      mean time: 0.088896 ms, speedup: 0.82
norm_fp32x4                    mean time: 0.043936 ms, speedup: 1.65
norm_fp32x4_split_k            mean time: 0.026400 ms, speedup: 2.75
####################################################################################################
n: 16, m: 8192
torch                          mean time: 0.011344 ms
norm_fp32                      mean time: 0.011376 ms, speedup: 1.00
norm_fp32x4                    mean time: 0.009568 ms, speedup: 1.19
norm_fp32x4_split_k            mean time: 0.013504 ms, speedup: 0.84
####################################################################################################
n: 16, m: 12800
torch                          mean time: 0.014032 ms
norm_fp32                      mean time: 0.013488 ms, speedup: 1.04
norm_fp32x4                    mean time: 0.013312 ms, speedup: 1.05
norm_fp32x4_split_k            mean time: 0.017440 ms, speedup: 0.80
####################################################################################################
n: 16, m: 102400
torch                          mean time: 0.079440 ms
norm_fp32                      mean time: 0.100480 ms, speedup: 0.79
norm_fp32x4                    mean time: 0.050576 ms, speedup: 1.57
norm_fp32x4_split_k            mean time: 0.041776 ms, speedup: 1.90
```

#### 全部版本测试

```bash
####################################################################################################
n: 64, m: 2048
torch                          mean time: 0.009440 ms
norm_fp32                      mean time: 0.009824 ms, speedup: 0.96
norm_fp32x4                    mean time: 0.011360 ms, speedup: 0.83
norm_fp16                      mean time: 0.011216 ms, speedup: 0.84
norm_fp16x8_packed             mean time: 0.010704 ms, speedup: 0.88
####################################################################################################
n: 64, m: 4096
torch                          mean time: 0.011040 ms
norm_fp32                      mean time: 0.013360 ms, speedup: 0.83
norm_fp32x4                    mean time: 0.013456 ms, speedup: 0.82
norm_fp16                      mean time: 0.013472 ms, speedup: 0.82
norm_fp16x8_packed             mean time: 0.011488 ms, speedup: 0.96
####################################################################################################
n: 64, m: 8192
torch                          mean time: 0.013696 ms
norm_fp32                      mean time: 0.021792 ms, speedup: 0.63
norm_fp32x4                    mean time: 0.019312 ms, speedup: 0.71
norm_fp16                      mean time: 0.019776 ms, speedup: 0.69
norm_fp16x8_packed             mean time: 0.013360 ms, speedup: 1.03
####################################################################################################
n: 64, m: 12800
torch                          mean time: 0.017776 ms
norm_fp32                      mean time: 0.031904 ms, speedup: 0.56
norm_fp32x4                    mean time: 0.021376 ms, speedup: 0.83
norm_fp16                      mean time: 0.025568 ms, speedup: 0.70
norm_fp16x8_packed             mean time: 0.014768 ms, speedup: 1.20
####################################################################################################
n: 128, m: 2048
torch                          mean time: 0.009536 ms
norm_fp32                      mean time: 0.008160 ms, speedup: 1.17
norm_fp32x4                    mean time: 0.011120 ms, speedup: 0.86
norm_fp16                      mean time: 0.009792 ms, speedup: 0.97
norm_fp16x8_packed             mean time: 0.008992 ms, speedup: 1.06
####################################################################################################
n: 128, m: 4096
torch                          mean time: 0.012976 ms
norm_fp32                      mean time: 0.014944 ms, speedup: 0.87
norm_fp32x4                    mean time: 0.014768 ms, speedup: 0.88
norm_fp16                      mean time: 0.012272 ms, speedup: 1.06
norm_fp16x8_packed             mean time: 0.011168 ms, speedup: 1.16
####################################################################################################
n: 128, m: 8192
torch                          mean time: 0.017920 ms
norm_fp32                      mean time: 0.029600 ms, speedup: 0.61
norm_fp32x4                    mean time: 0.026992 ms, speedup: 0.66
norm_fp16                      mean time: 0.019584 ms, speedup: 0.92
norm_fp16x8_packed             mean time: 0.014976 ms, speedup: 1.20
####################################################################################################
n: 128, m: 12800
torch                          mean time: 0.041424 ms
norm_fp32                      mean time: 0.049440 ms, speedup: 0.84
norm_fp32x4                    mean time: 0.049520 ms, speedup: 0.84
norm_fp16                      mean time: 0.034992 ms, speedup: 1.18
norm_fp16x8_packed             mean time: 0.021488 ms, speedup: 1.93
####################################################################################################
n: 512, m: 2048
torch                          mean time: 0.019104 ms
norm_fp32                      mean time: 0.025600 ms, speedup: 0.75
norm_fp32x4                    mean time: 0.025552 ms, speedup: 0.75
norm_fp16                      mean time: 0.019120 ms, speedup: 1.00
norm_fp16x8_packed             mean time: 0.014560 ms, speedup: 1.31
####################################################################################################
n: 512, m: 4096
torch                          mean time: 0.048064 ms
norm_fp32                      mean time: 0.056416 ms, speedup: 0.85
norm_fp32x4                    mean time: 0.054464 ms, speedup: 0.88
norm_fp16                      mean time: 0.036608 ms, speedup: 1.31
norm_fp16x8_packed             mean time: 0.030144 ms, speedup: 1.59
####################################################################################################
n: 512, m: 8192
torch                          mean time: 0.103648 ms
norm_fp32                      mean time: 0.106768 ms, speedup: 0.97
norm_fp32x4                    mean time: 0.103792 ms, speedup: 1.00
norm_fp16                      mean time: 0.062496 ms, speedup: 1.66
norm_fp16x8_packed             mean time: 0.053552 ms, speedup: 1.94
####################################################################################################
n: 512, m: 12800
torch                          mean time: 0.185216 ms
norm_fp32                      mean time: 0.169520 ms, speedup: 1.09
norm_fp32x4                    mean time: 0.165104 ms, speedup: 1.12
norm_fp16                      mean time: 0.094672 ms, speedup: 1.96
norm_fp16x8_packed             mean time: 0.084624 ms, speedup: 2.19
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.047648 ms
norm_fp32                      mean time: 0.054416 ms, speedup: 0.88
norm_fp32x4                    mean time: 0.052608 ms, speedup: 0.91
norm_fp16                      mean time: 0.033248 ms, speedup: 1.43
norm_fp16x8_packed             mean time: 0.025280 ms, speedup: 1.88
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.111424 ms
norm_fp32                      mean time: 0.111280 ms, speedup: 1.00
norm_fp32x4                    mean time: 0.110336 ms, speedup: 1.01
norm_fp16                      mean time: 0.060320 ms, speedup: 1.85
norm_fp16x8_packed             mean time: 0.053856 ms, speedup: 2.07
####################################################################################################
n: 1024, m: 8192
torch                          mean time: 0.211872 ms
norm_fp32                      mean time: 0.206912 ms, speedup: 1.02
norm_fp32x4                    mean time: 0.205552 ms, speedup: 1.03
norm_fp16                      mean time: 0.112928 ms, speedup: 1.88
norm_fp16x8_packed             mean time: 0.107008 ms, speedup: 1.98
####################################################################################################
n: 1024, m: 12800
torch                          mean time: 0.458320 ms
norm_fp32                      mean time: 0.350544 ms, speedup: 1.31
norm_fp32x4                    mean time: 0.344272 ms, speedup: 1.33
norm_fp16                      mean time: 0.184640 ms, speedup: 2.48
norm_fp16x8_packed             mean time: 0.164720 ms, speedup: 2.78
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.209008 ms
norm_fp32                      mean time: 0.207680 ms, speedup: 1.01
norm_fp32x4                    mean time: 0.206848 ms, speedup: 1.01
norm_fp16                      mean time: 0.108416 ms, speedup: 1.93
norm_fp16x8_packed             mean time: 0.105712 ms, speedup: 1.98
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.414272 ms
norm_fp32                      mean time: 0.411104 ms, speedup: 1.01
norm_fp32x4                    mean time: 0.405792 ms, speedup: 1.02
norm_fp16                      mean time: 0.211952 ms, speedup: 1.95
norm_fp16x8_packed             mean time: 0.207488 ms, speedup: 2.00
####################################################################################################
n: 4096, m: 8192
torch                          mean time: 0.820816 ms
norm_fp32                      mean time: 0.813520 ms, speedup: 1.01
norm_fp32x4                    mean time: 0.805072 ms, speedup: 1.02
norm_fp16                      mean time: 0.418432 ms, speedup: 1.96
norm_fp16x8_packed             mean time: 0.406032 ms, speedup: 2.02
####################################################################################################
n: 4096, m: 12800
torch                          mean time: 1.444640 ms
norm_fp32                      mean time: 1.353856 ms, speedup: 1.07
norm_fp32x4                    mean time: 1.358960 ms, speedup: 1.06
norm_fp16                      mean time: 0.691024 ms, speedup: 2.09
norm_fp16x8_packed             mean time: 0.673696 ms, speedup: 2.14
```
