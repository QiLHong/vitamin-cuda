# hgemm

[English](./readme.en.md) · [中文入门](../../docs/zh-CN/入门教程.md)

> 本页翻译原文并补充源码导读；功能勾选沿用原作者的实现状态，性能数据为历史记录，本次未复测。

## 概览

半精度矩阵乘法（GEMM）核函数，支持 BF16 和 FP16。

- [x] hgemm_cublas — BF16/FP16，cuBLAS 基准实现
- [x] hgemm_naive — BF16/FP16，使用 ldmatrix 加载矩阵片段并执行 MMA
- [x] hgemm_bcf — BF16/FP16，ldmatrix + MMA；对 As/Bs 做 swizzle 重排以避免 Bank 冲突（BCF）；原文报告性能约为 cuBLAS 的 95%～99%
- [x] hgemm_bcf_dbf — BF16/FP16，在 ldmatrix + MMA 和 BCF 重排基础上加入双缓冲；原文报告快于 cuBLAS
- [x] hgemm_bcf_dbf_rw — BF16/FP16，ldmatrix + MMA、BCF、双缓冲和合并的全局内存读写；原文报告快于 cuBLAS
- [x] PyTorch 算子绑定与结果差异检查

## 中文导读与使用前提

本目录计算行主序矩阵 `C[M,N] = A[M,K] @ B[K,N]`。`naive` 在这里仍然使用 Tensor Core，不是“一线程一个输出”的朴素 FP32 实现。每块 256 个线程分成 8 个线程束，共计算 128×128 个输出元素；每线程保存 64 个 FP32 累加值，最后再转成 FP16 或 BF16。

阅读顺序建议为 `hgemm_naive` → `hgemm_bcf` → `hgemm_bcf_dbf` → `hgemm_bcf_dbf_rw`：先理解片段加载和 MMA，再看共享内存地址重排、交替读写的两组缓冲，以及复用共享内存来合并结果写回。`cp.async.wait_group` 等待异步拷贝，`__syncthreads()` 协调线程块内的访问，两者承担不同职责。

从当前索引可推得，入门使用应限定为正尺寸、`M` 为 128 的倍数、`N` 为 1024 的倍数、`K` 为 32 的倍数。`N` 的要求不仅来自 128 列分块，还来自宽度为 8 的网格重排；末尾不足 8 个列块没有独立保护。宽加载与输出也没有逐元素尾部检查。输入输出须同设备、同 dtype、连续、形状匹配且无有害重叠，相关地址还须满足 16 字节宽访问对齐。包装只检查部分条件，非 FP16 的分支直接按 BF16 解释数据，不能传任意 dtype。这些使用前提来自源码静态阅读，不代表所有满足条件的输入都已经实测。

`test.py` 默认执行 BF16、4096³ 的比较；FP16 循环和 `test_all()` 仍处于原有的注释状态。计时使用主机时钟包围多次调用，并在前后同步，输出的是总耗时除以重复次数，包含该调用方式下的调度开销。`test_for_ncu.py` 供 Nsight Compute 采集，默认只调用一次、没有预热；其中 `baseline=1e-6` 是占位值，未先测 PyTorch 时打印的 `speedup` 不具备对比意义。

以下命令在本目录执行，需要可用的 CUDA、PyTorch 扩展编译工具链及支持相关 PTX 指令的 GPU。`cp.async` 和本文件所用的 FP16/BF16 MMA 形状要求 SM80 或更高的相应指令支持，工具链也须支持实际设备。参见 [NVIDIA PTX 指令文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)。本次没有 CUDA 编译或 GPU 运行结果。

## 运行测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

## 历史示例输出

```yaml
####################################################################################################
n: 4096, m: 4096, k: 4096
torch                                    mean time: 4.097551 ms, 33.54 tflops
hgemm_cublas                             mean time: 4.210246 ms, speedup: 0.97, tflops: 32.64
hgemm_naive                              mean time: 5.191345 ms, speedup: 0.79, tflops: 26.47
hgemm_bcf                                mean time: 4.336920 ms, speedup: 0.94, tflops: 31.69
hgemm_bcf_dbf                            mean time: 4.096174 ms, speedup: 1.00, tflops: 33.55
hgemm_bcf_dbf_rw                         mean time: 4.075860 ms, speedup: 1.01, tflops: 33.72
```
