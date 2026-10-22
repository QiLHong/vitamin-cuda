# Warp 之间共享内存 Bank 冲突的观察实验

[中文首页](../../README.md) · [英文原文](./readme.en.md) · [原有中文文章](./readme_cn.md)

英文原始文件已完整保留，原有 `readme_cn.md` 也未修改。先切换到本目录再运行以下命令。

## 功能概览

这是一个用于观察共享内存 Bank 冲突计数的微基准；原文关注的解释是 **Warp 之间的冲突**。

- [x] `load_fp32x4` 核函数
- [x] PyTorch 算子绑定与正确性检查（沿用原清单；当前脚本实际没有结果断言）

这段代码把数据搬入、搬出共享内存（SMEM），中间只做很少的计算。原文认为，常见文章主要解释同一 Warp 内的 Bank 冲突，而较少讨论**多个 Warp 同时发出请求时出现的冲突计数**。下文保留这一实验叙述；对计数原因的解释需要与具体架构和工具语义区分。

原文的解释是：现代 GPU 的每个 SM 有多个子分区调度器（文中以常见的 4 个为例），多个 Warp 可同时发起 L1/TEX/SMEM 请求；当它们访问同一 Bank 的不同地址时，会产生这里观察到的冲突。原作者认为这种底层调度现象难以完全消除，可以尝试用向量化访存或 `ldmatrix` 减少请求分散，再通过异步拷贝与计算重叠隐藏开销。这是原文的实验解释，并非本次翻译新验证的硬件结论。

核函数使用 `float4` 向量化读写共享内存。按原文对该访问模式的分析，每个 Warp 需要 4 个 wavefront，预期**没有同一 Warp 内的冲突**。原文列出的三组不同规模历史实验中，前两组冲突计数为 0，较大规模出现非零计数。

具体而言，下方第三组历史输出记录了 266,148 个 wavefront 和 4,004 次冲突，二者比值约为 1.5%。

## 构建与采集

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
bash run_ncu.sh
```

### 原始 NCU 输出（非本次实测）

```
   load_fp32x4_kernel(float *, float *, int) (1, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                        0
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                           16
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                           16
    smsp__inst_executed_op_shared_ld.sum                            inst            4
    smsp__inst_executed_op_shared_st.sum                            inst            4
    -------------------------------------------------------- ----------- ------------

  load_fp32x4_kernel(float *, float *, int) (128, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                        0
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                         2048
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                         2048
    smsp__inst_executed_op_shared_ld.sum                            inst          512
    smsp__inst_executed_op_shared_st.sum                            inst          512
    -------------------------------------------------------- ----------- ------------

  load_fp32x4_kernel(float *, float *, int) (16384, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                     4004
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                        0
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                       266148
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                       262144
    smsp__inst_executed_op_shared_ld.sum                            inst        65536
    smsp__inst_executed_op_shared_st.sum                            inst        65536
    -------------------------------------------------------- ----------- ------------
```

原文推荐的 NVIDIA GTC 技术报告：<https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41723/>

## 译注：区分访问模式与计数器结论

[CUDA 编程指南的共享内存说明](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)将经典 Bank 冲突描述为同一 Warp 内多个线程访问同一 Bank 的不同元素。原文对跨 Warp 计数的解释不应代替这一定义；仅凭一个非零硬件计数器也不足以判定产生它的全部原因，应结合架构、生成指令、NCU 指标定义与理想 wavefront 数分析。

本核每个线程读取四个 FP32，只把第一个分量乘 2，然后经共享内存写回。块固定为 128 线程，共享数组为 512 个 FP32；网格用整数除法，核内没有尾部保护，原输入均为正的 512 倍数。还需满足连续 FP32、足够容量、同设备及 `float4` 对齐。`test_for_ncu.py` 只启动核函数，没有 `assert` 或误差对比，原功能清单不等于完成正确性测试。

本次没有执行 NCU、CUDA 编译或 GPU 测试。指标可用性和采集权限也应在实际环境中确认。
