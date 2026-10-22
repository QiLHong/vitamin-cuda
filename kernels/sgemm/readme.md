# SGEMM

[English](./readme.en.md) · [中文入门](../../docs/zh-CN/入门教程.md)

> 本页翻译原文并补充源码导读；功能勾选沿用原作者的实现状态，性能数据为历史记录，本次未复测。

## 概览

### SGEMM 的 SIMT 核函数

- [x] `sgemm_cublas` — cuBLAS 的 FP32 基准实现
- [x] `sgemm_tiling` — 向量化读写与基于共享内存的线程块分块
- [x] `sgemm_at_tiling` — 向量化读写，并将 A 转置后存入共享内存（写入存在四路 Bank 冲突，内层循环用 `float4` 读取）
- [x] `sgemm_at_bcf_swizzling` — 向量化读写、A 转置与 swizzle 重排，针对该访问布局消除 Bank 冲突
- [x] `sgemm_at_bcf_swizzling_rw` — 进一步合并 C 的写回访存
- [x] `sgemm_at_bcf_swizzling_dbf_rw` — 进一步加入双缓冲流水线；**原文报告性能超过 cuBLAS**
- [x] PyTorch 算子绑定与正确性检查

### SGEMM 的 TF32 Tensor Core 核函数

- [x] `sgemm_cublas_tf32` — cuBLAS 的 TF32 基准实现
- [x] `sgemm_tf32_bt` — 向量化加载 A/B，将 B 转置后写入共享内存，再使用 `ldmatrix` + `mma`
- [x] `sgemm_tf32_bt_swizzle` — 进一步消除 A 在共享内存中的 Bank 冲突
- [x] `sgemm_tf32_bt_swizzle_dbf` — 进一步加入网格 swizzle；原文报告性能为 cuBLAS 的 97%～102%
- [x] `sgemm_tf32_swizzle_bcf` — 使用 `cp.async` 加载 A/B、线程束 shuffle 对 B 的寄存器片段做转置、消除 A/B 共享内存 Bank 冲突，并重排网格（此项沿用原文概述，当前源码差异见下方译注）
- [x] `sgemm_tf32_swizzle_bcf_dbf` — 进一步加入双缓冲；**原文报告性能超过 cuBLAS**
- [x] PyTorch 算子绑定与正确性检查

## 中文导读与使用前提

本目录计算 `C[M,N] = A[M,K] @ B[K,N]`，输入输出都是连续行主序 FP32 张量。SIMT 分块版本让每个线程累加一个 8×8 输出片段；TF32 版本用 `mma.sync` 的矩阵片段布局组织 Tensor Core 计算。TF32 运算的精度含义与完整 FP32 运算不同，不能因 Tensor 的存储 dtype 仍为 FP32 就认为两条路径数值等价。测试中 SIMT 的默认容差为 `1e-3`，包含 `tf32` 的名称则用 `0.1`，这只是原脚本的检查设置。

译注：原实现把加载或位重解释得到的值交给 TF32 MMA，没有显式执行 FP32→TF32 数值转换。搬运与位重解释本身不等于格式转换；这一路径的格式约定和数值行为需单独核实。PTX 对 TF32 的内部布局采用实现相关约定，并提供从 FP32 转为 TF32 的指令，参见 [PTX 浮点数据格式](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#alternate-floating-point-data-formats)。本次不修改该计算路径。

可按 `sgemm_naive` → `sgemm_tiling` → `sgemm_at_tiling` → `sgemm_at_bcf_swizzling` → `..._rw` → `..._dbf_rw` 阅读：从每线程一个输出，逐步引入共享内存复用、A 的转置、地址重排、合并写回和双缓冲。原功能清单没有列出的 `sgemm_naive` 仍在源码中绑定。

朴素核函数有行列边界判断；分块核函数没有覆盖任意尾部，包装中的网格向上取整也不能替代核内保护。SIMT 分块版本要求按当前实现使用 `M/N` 为 128 的正整数倍、`K` 为 16 的正整数倍，并满足 `float4` 的 16 字节对齐。TF32 版本同样使用 128×128×16 分块，其中 `sgemm_tf32_bt_swizzle_dbf`、`sgemm_tf32_swizzle_bcf`、`sgemm_tf32_swizzle_bcf_dbf` 还采用宽度为 8 的网格重排，入门使用 `N` 为 1024 的倍数。输入输出须同设备、形状匹配且无有害重叠，当前包装没有逐项保证这些条件。

译注：原清单对 `sgemm_tf32_swizzle_bcf` 写有“warp-shuffle B-register transpose”。当前文件对应版本按 `b_row_0/b_row_1` 直接读取 B 的共享内存片段，没有调用线程束 shuffle 指令；理解执行行为时应以源码为准。Bank 冲突和性能描述也只针对原文讨论的特定访问布局与实验条件。

`test.py` 默认只调用 `test_tf32_4096()`；SIMT 的 `test_4096()` 和全尺寸的 `test_all()` 保留为未启用入口。计时采用主机时钟与前后同步，报告重复调用的平均耗时。`test_for_ncu.py` 用于单次采集，默认无预热，且 PyTorch 基准被注释掉；此时 `baseline=1e-6` 生成的 `speedup` 没有有效对照含义。`run_ncu.sh` 需要在本目录执行，会按核函数名称筛选采集并覆盖同名报告。

以下所有性能输出都保留原始字节、名称和数字，可能反映早期实现。它们不是本次翻译后的实测结果，也不保证在其他 GPU、编译器、矩阵形状或计时方法下重现。测试依赖 CUDA/PyTorch 编译环境；扩展包含 TF32 的 `mma` 与 `cp.async` 路径，需要支持相应指令的目标架构（这些指令形态至少要求 SM80）。参见 [NVIDIA PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)。本次未编译或运行 GPU 测试。

## 编译与测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

## SGEMM TF32 历史基准结果

```yaml
####################################################################################################
n: 4096, m: 4096, k: 4096
torch                                    mean time: 15.146454 ms, 9.07 tflops
sgemm_cublas_tf32                        mean time: 8.535476 ms, speedup: 1.77, tflops: 16.10
sgemm_tf32_bt                            mean time: 15.925327 ms, speedup: 0.95, tflops: 8.63
sgemm_tf32_bt_swizzle                    mean time: 9.786451 ms, speedup: 1.55, tflops: 14.04
sgemm_tf32_bt_swizzle_dbf                mean time: 8.723189 ms, speedup: 1.83, tflops: 15.76
sgemm_tf32_swizzle_bcf                   mean time: 8.650843 ms, speedup: 1.83, tflops: 15.89
sgemm_tf32_swizzle_bcf_dbf               mean time: 8.275736 ms, speedup: 1.92, tflops: 16.61
```

## SGEMM SIMT 历史基准结果

### 4096×4096×4096

```bash
####################################################################################################
n: 4096, m: 4096, k: 4096
torch                          mean time: 14.974799 ms
sgemm_cublas                   mean time: 14.523163 ms, speedup: 1.03
sgemm_tiling                   mean time: 18.760985 ms, speedup: 0.80
sgemm_at_tiling                mean time: 16.436968 ms, speedup: 0.91
sgemm_at_bcf_swizzling         mean time: 15.706529 ms, speedup: 0.95
sgemm_at_bcf_swizzling_rw      mean time: 15.522802 ms, speedup: 0.96
sgemm_at_bcf_swizzling_dbf_rw  mean time: 14.193397 ms, speedup: 1.06
sgemm_cublas_tf32              mean time:  8.798057 ms, speedup: 1.70
```

### 各种矩阵规模

```bash
####################################################################################################
m: 512, n: 256, k: 256
torch                          mean time: 0.062467 ms
sgemm_cublas                   mean time: 0.040665 ms, speedup: 1.54
sgemm                          mean time: 0.033703 ms, speedup: 1.85
sgemm_bcf                      mean time: 0.031690 ms, speedup: 1.97
sgemm_cublas_tf32              mean time: 0.012136 ms, speedup: 5.15
####################################################################################################
m: 512, n: 512, k: 256
torch                          mean time: 0.026130 ms
sgemm_cublas                   mean time: 0.025757 ms, speedup: 1.01
sgemm                          mean time: 0.033650 ms, speedup: 0.78
sgemm_bcf                      mean time: 0.057856 ms, speedup: 0.45
sgemm_cublas_tf32              mean time: 0.018859 ms, speedup: 1.39
####################################################################################################
m: 512, n: 256, k: 512
torch                          mean time: 0.056480 ms
sgemm_cublas                   mean time: 0.031005 ms, speedup: 1.82
sgemm                          mean time: 0.058627 ms, speedup: 0.96
sgemm_bcf                      mean time: 0.062091 ms, speedup: 0.91
sgemm_cublas_tf32              mean time: 0.020900 ms, speedup: 2.70
####################################################################################################
m: 512, n: 512, k: 512
torch                          mean time: 0.058759 ms
sgemm_cublas                   mean time: 0.038631 ms, speedup: 1.52
sgemm                          mean time: 0.086716 ms, speedup: 0.68
sgemm_bcf                      mean time: 0.055433 ms, speedup: 1.06
sgemm_cublas_tf32              mean time: 0.025164 ms, speedup: 2.34
####################################################################################################
m: 1024, n: 512, k: 512
torch                          mean time: 0.072586 ms
sgemm_cublas                   mean time: 0.070146 ms, speedup: 1.03
sgemm                          mean time: 0.105271 ms, speedup: 0.69
sgemm_bcf                      mean time: 0.116874 ms, speedup: 0.62
sgemm_cublas_tf32              mean time: 0.046662 ms, speedup: 1.56
####################################################################################################
m: 1024, n: 1024, k: 512
torch                          mean time: 0.152143 ms
sgemm_cublas                   mean time: 0.149413 ms, speedup: 1.02
sgemm                          mean time: 0.188179 ms, speedup: 0.81
sgemm_bcf                      mean time: 0.143108 ms, speedup: 1.06
sgemm_cublas_tf32              mean time: 0.075008 ms, speedup: 2.03
####################################################################################################
m: 1024, n: 512, k: 1024
torch                          mean time: 0.152300 ms
sgemm_cublas                   mean time: 0.158801 ms, speedup: 0.96
sgemm                          mean time: 0.225028 ms, speedup: 0.68
sgemm_bcf                      mean time: 0.208690 ms, speedup: 0.73
sgemm_cublas_tf32              mean time: 0.095371 ms, speedup: 1.60
####################################################################################################
m: 1024, n: 1024, k: 1024
torch                          mean time: 0.311959 ms
sgemm_cublas                   mean time: 0.280129 ms, speedup: 1.11
sgemm                          mean time: 0.343944 ms, speedup: 0.91
sgemm_bcf                      mean time: 0.327775 ms, speedup: 0.95
sgemm_cublas_tf32              mean time: 0.166107 ms, speedup: 1.88
####################################################################################################
m: 4096, n: 1024, k: 1024
torch                          mean time: 1.042517 ms
sgemm_cublas                   mean time: 1.080616 ms, speedup: 0.96
sgemm                          mean time: 1.178622 ms, speedup: 0.88
sgemm_bcf                      mean time: 1.149699 ms, speedup: 0.91
sgemm_cublas_tf32              mean time: 0.634827 ms, speedup: 1.64
####################################################################################################
m: 4096, n: 4096, k: 1024
torch                          mean time: 4.542118 ms
sgemm_cublas                   mean time: 4.757253 ms, speedup: 0.95
sgemm                          mean time: 4.549831 ms, speedup: 1.00
sgemm_bcf                      mean time: 4.394182 ms, speedup: 1.03
sgemm_cublas_tf32              mean time: 2.476518 ms, speedup: 1.83
####################################################################################################
m: 4096, n: 1024, k: 4096
torch                          mean time: 4.107055 ms
sgemm_cublas                   mean time: 4.653569 ms, speedup: 0.88
sgemm                          mean time: 4.289510 ms, speedup: 0.96
sgemm_bcf                      mean time: 4.529149 ms, speedup: 0.91
sgemm_cublas_tf32              mean time: 2.390973 ms, speedup: 1.72
####################################################################################################
m: 4096, n: 4096, k: 4096
torch                          mean time: 16.741930 ms
sgemm_cublas                   mean time: 16.232075 ms, speedup: 1.03
sgemm                          mean time: 17.583192 ms, speedup: 0.95
sgemm_bcf                      mean time: 16.882191 ms, speedup: 0.99
sgemm_cublas_tf32              mean time: 8.862989 ms, speedup: 1.89
####################################################################################################
m: 8192, n: 4096, k: 4096
torch                          mean time: 28.809838 ms
sgemm_cublas                   mean time: 29.222592 ms, speedup: 0.99
sgemm                          mean time: 34.324652 ms, speedup: 0.84
sgemm_bcf                      mean time: 32.647704 ms, speedup: 0.88
sgemm_cublas_tf32              mean time: 17.683384 ms, speedup: 1.63
####################################################################################################
m: 8192, n: 8192, k: 4096
torch                          mean time: 58.028348 ms
sgemm_cublas                   mean time: 57.443838 ms, speedup: 1.01
sgemm                          mean time: 70.755544 ms, speedup: 0.82
sgemm_bcf                      mean time: 69.478713 ms, speedup: 0.84
sgemm_cublas_tf32              mean time: 37.227454 ms, speedup: 1.56
####################################################################################################
m: 8192, n: 4096, k: 8192
torch                          mean time: 72.159470 ms
sgemm_cublas                   mean time: 73.127017 ms, speedup: 0.99
sgemm                          mean time: 76.206392 ms, speedup: 0.95
sgemm_bcf                      mean time: 73.002593 ms, speedup: 0.99
sgemm_cublas_tf32              mean time: 37.799734 ms, speedup: 1.91
####################################################################################################
m: 8192, n: 8192, k: 8192
torch                          mean time: 127.732361 ms
sgemm_cublas                   mean time: 125.815921 ms, speedup: 1.02
sgemm                          mean time: 153.963272 ms, speedup: 0.83
sgemm_bcf                      mean time: 148.995780 ms, speedup: 0.86
sgemm_cublas_tf32              mean time: 78.690780 ms, speedup: 1.62
```
