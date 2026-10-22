# Softmax（归一化指数函数）

[英文原文](./readme.en.md) · [中文首页](../../README.md) · [翻译进度](../../docs/zh-CN/翻译进度.md)

> 本页翻译并保留原始章节、命令与历史输出；本次仅修改文档和源码注释，未进行 CUDA 编译、GPU 正确性测试或性能复测。

## 概述

使用最大值平移的在线 softmax 核函数，针对不同行宽提供多种实现策略。原文中的 “safe” 指指数计算的数值稳定性设计，不表示对任意输入形状或 NaN/Inf 都有完整支持。

- [x] 单遍读取方案（每行一个线程块，使用寄存器或寄存器与共享内存缓存）
  - [x] 在线 softmax：FP32 / FP16
  - [x] 在线 softmax：FP32 向量化（×4）
  - [x] 在线 softmax：FP16 向量化（×8，仅用寄存器，打包读写）
  - [x] 在线 softmax **medium**：FP16 向量化（适量寄存器 + 共享内存，打包读写）
  - [x] 在线 softmax **extreme**：FP16 向量化（更多寄存器 + 共享内存，打包读写）
- [x] 两遍读取方案（面向超出单块固定缓存容量的大行宽）
  - [x] 在线 softmax **arbitrary**：FP16 向量化、打包读写；当前源码分两遍读取输入，不缓存整行（原概述写作“最大寄存器 + SMEM”，与当前实现不完全对应）
  - [x] 在线 softmax **split-k**：FP16 向量化、打包读写；分块统计后借助全局工作区合并（原概述写作“最大寄存器 + SMEM”）
- [x] PyTorch 算子绑定与正确性检查

## 中文阅读指引与容量限制

输入/输出布局为连续二维 `[bs,hidden_size]`。先读 [CUDA 实现](./softmax.cu) 的 `MD` 与 `block_online_softmax_reduce`：每组数据记录 `(m,d)`，其中 `m` 是最大值，`d=sum(exp(x-m))`；合并两组时先统一最大值，再缩放各自的分母。这解释了在线合并为何不能直接把两个 `d` 相加。

下面的容量由当前模板参数与循环次数计算，**不是本次 GPU 测试结果**。包装没有自动检查所有容量或尾部条件。

| Python 入口 | 输入类型与单行容量 | 调用方需关注的条件 |
| --- | --- | --- |
| `softmax` | FP32/FP16，最多 8192 列 | 每线程最多缓存 32 个标量；超出的列不会被处理 |
| `softmax_fp32x4` | FP32，最多 8192 列 | 行宽为 4 的倍数，首地址及每行满足 16 字节对齐 |
| `softmax_fp16x8_packed` | FP16，最多 32768 列 | 每线程 16 组，每组 8 元素；原注释的 8192 与当前循环不符 |
| `softmax_medium` | FP16，最多 32768 列 | `(8+8)*256*8`，动态共享内存 32768 字节 |
| `softmax_extreme` | FP16，最多 114688 列 | `(32+24)*256*8`，动态共享内存 98304 字节，受设备资源限制 |
| `softmax_arbitrary` | FP16，两遍读取，未用固定数组缓存整行 | 名称不代表任意 dtype/形状；仍按 8 元素整组读写 |
| `softmax_splitk` | FP16，每块分片容量 32768 列 | 第一阶段统计，第二阶段合并并写回；使用 FP32 全局工作区 |

所有 FP16 打包路径都要求行宽为 8 的倍数及相应的 16 字节对齐，因为判断只检查数据组起点。`medium`、`extreme`、`arbitrary`、`splitk` 的包装直接把指针解释为 `half*`，不会转换错误 dtype。输入输出还应同形、同设备且空间充足；空输入、超大整数索引、NaN/Inf 与输入输出重叠需要另行分析。

`test.py` 默认只调用 `test_small()`，部分 FP32 测试在原文件中已注释；不能把脚本成功运行理解为所有入口已测。其计时取中位数，保留的输出标签仍为 `mean time`。`test_for_ncu.py`/`run_ncu.sh` 用于分析器采集，与普通计时条件不同。

## 编译与测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 历史 PTX 编译输出

```yaml
ptxas info    : 28 bytes gmem
ptxas info    : Compiling entry function '_Z18softmax_grid_pass2ILi256ELi16EEvP6__halfS1_PfS2_i' for 'sm_120'
ptxas info    : Function properties for _Z18softmax_grid_pass2ILi256ELi16EEvP6__halfS1_PfS2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 84.659 ms
ptxas info    : Compiling entry function '_Z18softmax_grid_pass1ILi256ELi16EEvP6__halfPfS2_i' for 'sm_120'
ptxas info    : Function properties for _Z18softmax_grid_pass1ILi256ELi16EEvP6__halfPfS2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 79 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 21.369 ms
ptxas info    : Compiling entry function '_Z24softmax_arbitrary_kernelILi256EEvP6__halfS1_i' for 'sm_120'
ptxas info    : Function properties for _Z24softmax_arbitrary_kernelILi256EEvP6__halfS1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 36 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 12.731 ms
ptxas info    : Compiling entry function '_Z22softmax_onepass_kernelILi256ELi32ELi24ELi1EEvP6__halfS1_i' for 'sm_120'
ptxas info    : Function properties for _Z22softmax_onepass_kernelILi256ELi32ELi24ELi1EEvP6__halfS1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 250 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 176.764 ms
ptxas info    : Compiling entry function '_Z22softmax_onepass_kernelILi256ELi8ELi8ELi3EEvP6__halfS1_i' for 'sm_120'
ptxas info    : Function properties for _Z22softmax_onepass_kernelILi256ELi8ELi8ELi3EEvP6__halfS1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 80 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 37.731 ms
ptxas info    : Compiling entry function '_Z28softmax_fp16x8_packed_kernelILi256EEvP6__halfS1_i' for 'sm_120'
ptxas info    : Function properties for _Z28softmax_fp16x8_packed_kernelILi256EEvP6__halfS1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 95 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 38.098 ms
ptxas info    : Compiling entry function '_Z21softmax_fp32x4_kernelILi256EEvPfS0_i' for 'sm_120'
ptxas info    : Function properties for _Z21softmax_fp32x4_kernelILi256EEvPfS0_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 55 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 10.820 ms
ptxas info    : Compiling entry function '_Z14softmax_kernelILi256E6__halfEvPT0_S2_i' for 'sm_120'
ptxas info    : Function properties for _Z14softmax_kernelILi256E6__halfEvPT0_S2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 90 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 20.868 ms
ptxas info    : Compiling entry function '_Z14softmax_kernelILi256EfEvPT0_S1_i' for 'sm_120'
ptxas info    : Function properties for _Z14softmax_kernelILi256EfEvPT0_S1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 86 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 17.167 ms
```

### 历史基准结果

#### 大行宽

```yaml
####################################################################################################
bs: 4, hidden_size: 16384
torch                          mean time: 0.010752 ms
softmax_medium                 mean time: 0.009040 ms, speedup: 1.19
softmax_extreme                mean time: 0.013120 ms, speedup: 0.82
softmax_arbitrary              mean time: 0.009216 ms, speedup: 1.17
softmax_splitk                 mean time: 0.014928 ms, speedup: 0.72
####################################################################################################
bs: 4, hidden_size: 32768
torch                          mean time: 0.013216 ms
softmax_medium                 mean time: 0.011072 ms, speedup: 1.19
softmax_extreme                mean time: 0.014928 ms, speedup: 0.89
softmax_arbitrary              mean time: 0.012992 ms, speedup: 1.02
softmax_splitk                 mean time: 0.017328 ms, speedup: 0.76
####################################################################################################
bs: 4, hidden_size: 65536
torch                          mean time: 0.019792 ms
softmax_extreme                mean time: 0.017712 ms, speedup: 1.12
softmax_arbitrary              mean time: 0.021504 ms, speedup: 0.92
softmax_splitk                 mean time: 0.017872 ms, speedup: 1.11
####################################################################################################
bs: 4, hidden_size: 114688
torch                          mean time: 0.029744 ms
softmax_extreme                mean time: 0.021392 ms, speedup: 1.39
softmax_arbitrary              mean time: 0.031872 ms, speedup: 0.93
softmax_splitk                 mean time: 0.020512 ms, speedup: 1.45
####################################################################################################
bs: 4, hidden_size: 262144
torch                          mean time: 0.061904 ms
softmax_arbitrary              mean time: 0.064176 ms, speedup: 0.96
softmax_splitk                 mean time: 0.023168 ms, speedup: 2.67
####################################################################################################
bs: 4, hidden_size: 1048576
torch                          mean time: 0.202112 ms
softmax_arbitrary              mean time: 0.233776 ms, speedup: 0.86
softmax_splitk                 mean time: 0.054720 ms, speedup: 3.69
####################################################################################################
bs: 4, hidden_size: 8388608
torch                          mean time: 1.730880 ms
softmax_arbitrary              mean time: 2.568016 ms, speedup: 0.67
softmax_splitk                 mean time: 0.603280 ms, speedup: 2.87
####################################################################################################
bs: 4, hidden_size: 33554432
torch                          mean time: 6.900256 ms
softmax_arbitrary              mean time: 10.154320 ms, speedup: 0.68
softmax_splitk                 mean time: 2.340544 ms, speedup: 2.95
```

#### 小行宽

```yaml
ninja: no work to do.
####################################################################################################
bs: 128, hidden_size: 1024
torch                          mean time: 0.011968 ms
softmax_fp16                   mean time: 0.013216 ms, speedup: 0.91
softmax_fp16x8_packed          mean time: 0.012608 ms, speedup: 0.95
softmax_medium                 mean time: 0.010784 ms, speedup: 1.11
softmax_extreme                mean time: 0.035936 ms, speedup: 0.33
softmax_arbitrary              mean time: 0.008704 ms, speedup: 1.37
####################################################################################################
bs: 128, hidden_size: 2048
torch                          mean time: 0.016896 ms
softmax_fp16                   mean time: 0.015680 ms, speedup: 1.08
softmax_fp16x8_packed          mean time: 0.013216 ms, speedup: 1.28
softmax_medium                 mean time: 0.011296 ms, speedup: 1.50
softmax_extreme                mean time: 0.038096 ms, speedup: 0.44
softmax_arbitrary              mean time: 0.008880 ms, speedup: 1.90
####################################################################################################
bs: 128, hidden_size: 4096
torch                          mean time: 0.019776 ms
softmax_fp16                   mean time: 0.019328 ms, speedup: 1.02
softmax_fp16x8_packed          mean time: 0.015968 ms, speedup: 1.24
softmax_medium                 mean time: 0.012912 ms, speedup: 1.53
softmax_extreme                mean time: 0.039920 ms, speedup: 0.50
softmax_arbitrary              mean time: 0.010832 ms, speedup: 1.83
####################################################################################################
bs: 128, hidden_size: 8192
torch                          mean time: 0.029472 ms
softmax_fp16                   mean time: 0.027776 ms, speedup: 1.06
softmax_fp16x8_packed          mean time: 0.018192 ms, speedup: 1.62
softmax_medium                 mean time: 0.015424 ms, speedup: 1.91
softmax_extreme                mean time: 0.041616 ms, speedup: 0.71
softmax_arbitrary              mean time: 0.014848 ms, speedup: 1.98
####################################################################################################
bs: 256, hidden_size: 1024
torch                          mean time: 0.013280 ms
softmax_fp16                   mean time: 0.019040 ms, speedup: 0.70
softmax_fp16x8_packed          mean time: 0.016832 ms, speedup: 0.79
softmax_medium                 mean time: 0.014528 ms, speedup: 0.91
softmax_extreme                mean time: 0.064000 ms, speedup: 0.21
softmax_arbitrary              mean time: 0.009536 ms, speedup: 1.39
####################################################################################################
bs: 256, hidden_size: 2048
torch                          mean time: 0.019424 ms
softmax_fp16                   mean time: 0.022848 ms, speedup: 0.85
softmax_fp16x8_packed          mean time: 0.019552 ms, speedup: 0.99
softmax_medium                 mean time: 0.017184 ms, speedup: 1.13
softmax_extreme                mean time: 0.066656 ms, speedup: 0.29
softmax_arbitrary              mean time: 0.012016 ms, speedup: 1.62
####################################################################################################
bs: 256, hidden_size: 4096
torch                          mean time: 0.035696 ms
softmax_fp16                   mean time: 0.029792 ms, speedup: 1.20
softmax_fp16x8_packed          mean time: 0.023616 ms, speedup: 1.51
softmax_medium                 mean time: 0.017600 ms, speedup: 2.03
softmax_extreme                mean time: 0.069232 ms, speedup: 0.52
softmax_arbitrary              mean time: 0.014720 ms, speedup: 2.42
####################################################################################################
bs: 256, hidden_size: 8192
torch                          mean time: 0.052144 ms
softmax_fp16                   mean time: 0.045152 ms, speedup: 1.15
softmax_fp16x8_packed          mean time: 0.028864 ms, speedup: 1.81
softmax_medium                 mean time: 0.023712 ms, speedup: 2.20
softmax_extreme                mean time: 0.073760 ms, speedup: 0.71
softmax_arbitrary              mean time: 0.025648 ms, speedup: 2.03
####################################################################################################
bs: 1024, hidden_size: 1024
torch                          mean time: 0.023488 ms
softmax_fp16                   mean time: 0.054608 ms, speedup: 0.43
softmax_fp16x8_packed          mean time: 0.047728 ms, speedup: 0.49
softmax_medium                 mean time: 0.037696 ms, speedup: 0.62
softmax_extreme                mean time: 0.231200 ms, speedup: 0.10
softmax_arbitrary              mean time: 0.017600 ms, speedup: 1.33
####################################################################################################
bs: 1024, hidden_size: 2048
torch                          mean time: 0.059168 ms
softmax_fp16                   mean time: 0.066816 ms, speedup: 0.89
softmax_fp16x8_packed          mean time: 0.054352 ms, speedup: 1.09
softmax_medium                 mean time: 0.043840 ms, speedup: 1.35
softmax_extreme                mean time: 0.249744 ms, speedup: 0.24
softmax_arbitrary              mean time: 0.024720 ms, speedup: 2.39
####################################################################################################
bs: 1024, hidden_size: 4096
torch                          mean time: 0.102240 ms
softmax_fp16                   mean time: 0.089808 ms, speedup: 1.14
softmax_fp16x8_packed          mean time: 0.076880 ms, speedup: 1.33
softmax_medium                 mean time: 0.051488 ms, speedup: 1.99
softmax_extreme                mean time: 0.251840 ms, speedup: 0.41
softmax_arbitrary              mean time: 0.053200 ms, speedup: 1.92
####################################################################################################
bs: 1024, hidden_size: 8192
torch                          mean time: 0.168304 ms
softmax_fp16                   mean time: 0.139968 ms, speedup: 1.20
softmax_fp16x8_packed          mean time: 0.111104 ms, speedup: 1.51
softmax_medium                 mean time: 0.105824 ms, speedup: 1.59
softmax_extreme                mean time: 0.263888 ms, speedup: 0.64
softmax_arbitrary              mean time: 0.105680 ms, speedup: 1.59
####################################################################################################
bs: 2048, hidden_size: 1024
torch                          mean time: 0.035936 ms
softmax_fp16                   mean time: 0.101328 ms, speedup: 0.35
softmax_fp16x8_packed          mean time: 0.086832 ms, speedup: 0.41
softmax_medium                 mean time: 0.070768 ms, speedup: 0.51
softmax_extreme                mean time: 0.444448 ms, speedup: 0.08
softmax_arbitrary              mean time: 0.030448 ms, speedup: 1.18
####################################################################################################
bs: 2048, hidden_size: 2048
torch                          mean time: 0.092544 ms
softmax_fp16                   mean time: 0.121664 ms, speedup: 0.76
softmax_fp16x8_packed          mean time: 0.098736 ms, speedup: 0.94
softmax_medium                 mean time: 0.077184 ms, speedup: 1.20
softmax_extreme                mean time: 0.469632 ms, speedup: 0.20
softmax_arbitrary              mean time: 0.054768 ms, speedup: 1.69
####################################################################################################
bs: 2048, hidden_size: 4096
torch                          mean time: 0.191808 ms
softmax_fp16                   mean time: 0.161408 ms, speedup: 1.19
softmax_fp16x8_packed          mean time: 0.137248 ms, speedup: 1.40
softmax_medium                 mean time: 0.108080 ms, speedup: 1.77
softmax_extreme                mean time: 0.477984 ms, speedup: 0.40
softmax_arbitrary              mean time: 0.105664 ms, speedup: 1.82
####################################################################################################
bs: 2048, hidden_size: 8192
torch                          mean time: 0.322592 ms
softmax_fp16                   mean time: 0.249664 ms, speedup: 1.29
softmax_fp16x8_packed          mean time: 0.210896 ms, speedup: 1.53
softmax_medium                 mean time: 0.204560 ms, speedup: 1.58
softmax_extreme                mean time: 0.491744 ms, speedup: 0.66
softmax_arbitrary              mean time: 0.210160 ms, speedup: 1.53
```
