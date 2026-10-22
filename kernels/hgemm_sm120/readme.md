# hgemm_sm120

[English](./readme.en.md) · [中文入门](../../docs/zh-CN/入门教程.md)

> 本页翻译原文并补充源码导读；功能勾选沿用原作者的实现状态，性能数据为历史记录，本次未复测。

## 概览

面向 SM120（Blackwell 架构类别）目标的半精度 GEMM。

- [x] hgemm_cublas：cuBLAS 基准实现
- [x] hgemm_bcf_dbf_rw：对照版本，使用 cp.async、ldmatrix、mma、双缓冲及合并的全局内存读写
- [x] hgemm_k_stages：cp.async、ldmatrix、mma、2/3 阶段缓冲及合并的全局内存访问
- [x] hgemm_tma_r_k_stages_64：TMA 读取、ldmatrix、mma；3 阶段缓冲、128×128×64 分块、寄存器双缓冲；原文另标注“一个线程块（one block）”
- [x] hgemm_tma_r_k_stages_32：TMA 读取、ldmatrix、mma；3 阶段缓冲、128×128×32 分块、寄存器双缓冲；原文另标注“两个线程块（two blocks）”
- [ ] TMA 读写、ldmatrix、mma（尚未完成）
- [x] PyTorch 算子绑定与结果差异检查

## 中文导读与使用前提

本模块在 128×128 的输出分块上比较 `cp.async` 流水线与 TMA 读取流水线。TMA 版本用描述符表达矩阵形状和步长；`tid=0` 发起 A 与两个 B 分片的传输，所有线程束等待对应屏障阶段后计算。共享内存有 3 个流水线阶段，阶段内部还用两组寄存器交替加载和执行 MMA。`_64` 与 `_32` 的区别是 K 方向分块大小；两个入口都是每个线程块计算一个 128×128 输出块。原文的“one block / two blocks”不能直接当作已验证的每 SM 驻留块数，实际驻留量取决于共享内存、寄存器和硬件资源。

`A[M,K]`、`B[K,N]`、`C[M,N]` 须为同设备、同 dtype 的连续行主序张量；dtype 只应使用 FP16/BF16。当前包装对非 FP16 数据直接选择 BF16 路径，没有完整检查形状、同设备和同 dtype。核函数的宽读取与结果写回没有完整尾部保护；入门约定 `M` 为 128 的正整数倍、`N` 为 1024 的正整数倍，使宽度为 8 的网格重排完整覆盖列块。

K 方向还须满足每个版本的分块及预取前提：`hgemm_bcf_dbf_rw` 使用 `BK=32`；默认 `hgemm_k_stages` 为 3 阶段，序言先预取 2 块，使用 `K≥64` 且为 32 的倍数；TMA `_32` 使用 `K≥64` 且为 32 的倍数，TMA `_64` 使用 `K≥128` 且为 64 的倍数。TMA 描述符还受地址、步长和 swizzle 对齐约束，编码失败会触发原有检查。这些约束只说明当前源码的索引前提，不是正确性测试结论。

`mbarrier` 的到达计数与传输字节计数应分开理解：初始化期望到达数为 1，发起线程用 `arrive.expect_tx` 到达并登记字节数，拷贝完成后更新事务计数。`mbarrier_wait` 循环中的 `0x989680` 是 10,000,000 **纳秒**的等待时间提示，原英文注释写成了 cycles；等待返回未完成时，汇编循环会重试，因此它不是整个核函数的超时上限。参见 [NVIDIA PTX 文档：mbarrier 等待指令](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)。异步代理的内存可见性、屏障阶段复用和具体架构行为仍需编译与真实设备验证。

`test.py` 默认测试 BF16、4096³；计时是同步前后主机时钟测得的多次调用平均值。`test_for_ncu.py` 默认零次预热、一次调用，并用 `baseline=1e-6` 占位，未运行 PyTorch 基准时其加速比不可作为性能证据。链接参数还保留了原实现的 `/usr/local/cuda/lib64/stubs` 路径，CUDA 安装位置不同需要另行调整环境。

本目录面向 SM120；需要支持目标架构、TMA 和相应驱动 API 的工具链与设备，不能仅靠设置 `TORCH_CUDA_ARCH_LIST` 让旧设备获得这些能力。本次仅翻译与静态检查，没有在 SM120 上编译、运行或复测性能。

下方保留原作者的测试命令。锁频是原实验的可选环境设置，`3050`/`12001` 为示例 MHz 数值，须按实际设备支持范围选择；`-lgc`/`-lmc` 分别锁定核心/显存频率，`-rgc`/`-rmc` 恢复设置。运行测试不要求照搬这些数值。

## 运行测试

```bash
nvidia-smi -q -d SUPPORTED_CLOCKS
nvidia-smi -lgc 3050   # lock GPU core clock (example MHz)
nvidia-smi -lmc 12001  # lock memory clock (example MHz)
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
nvidia-smi -rgc  # reset GPU clocks
nvidia-smi -rmc  # reset memory clocks
```

原文说明：TMA + ldmatrix + MMA 核函数经调优后能够正确运行，并测得了有竞争力的性能。这是原作者的历史报告，本次翻译没有重新验证。

## 历史示例输出

```yaml
####################################################################################################
n: 4096, m: 4096, k: 4096
torch                                    mean time: 4.011336 ms, 34.26 tflops
hgemm_cublas                             mean time: 4.258727 ms, speedup: 0.94, tflops: 32.27
hgemm_bcf_dbf_rw                         mean time: 4.042202 ms, speedup: 0.99, tflops: 34.00
hgemm_k_stages                           mean time: 4.131343 ms, speedup: 0.97, tflops: 33.27
hgemm_tma_r_k_stages_64                  mean time: 4.287896 ms, speedup: 0.94, tflops: 32.05
hgemm_tma_r_k_stages_32                  mean time: 4.005909 ms, speedup: 1.00, tflops: 34.31
```
