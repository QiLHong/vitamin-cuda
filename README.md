<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/WingEdge777/CDN@main/images/vitamin_cuda/vw-logo.jpg" alt="Vitamin-CUDA" width="800">
</p>

# 🍎 Vitamin-CUDA · 中文学习分支

> **每天学一个核函数，离高性能更近一步。**
>
> 从零基础到进阶的 CUDA 动手学习路线。

[英文原文](./README.en.md) · [零基础教程](./docs/zh-CN/入门教程.md) · [术语对照](./docs/zh-CN/术语表.md) · [翻译进度与验证范围](./docs/zh-CN/翻译进度.md)

**分支说明：** 本分支为 `zh-cn-beginner`，以 `main` 的 `ca40342f985a95be615585e02c130e13040f0e02` 为翻译基线。原作者、上游链接和 MIT 许可证保留。中文化逐模块进行，**下面的 `[x]` 表示原项目已实现该功能，不表示该模块已经完成中文翻译**；具体覆盖范围见进度文档。原始算法、函数名、编译参数、测试尺寸与英文运行输出不因翻译而修改。

Vitamin-CUDA 汇集了原作者介绍的 **170 多个手工优化 CUDA 核函数及变体**，从面向初学者的基本线程索引与网格步进循环，到面向性能研究的异步 TMA 流水线，覆盖 SGEMM、HGEMM、Flash Attention 和 Flash Decoding 等方向。项目通过 PyTorch C++ 扩展连接 Python 与 CUDA，便于比较输出和性能。

这里不追求过度封装或黑盒抽象，而是直接观察计算如何映射到 GPU 硬件。原介绍中的“零开销集成”“跑满硬件”等表达是项目定位，不是对所有输入、显卡和软件环境的性能保证。

## 新手从哪里开始？

先读 [入门教程](./docs/zh-CN/入门教程.md)，再按下面顺序阅读已有中文注释的文件：

1. [`kernels/elementwise/readme.md`](./kernels/elementwise/readme.md)：向量加法是什么、如何运行、哪些输入有边界限制。
2. [`kernels/elementwise/elementwise.cu`](./kernels/elementwise/elementwise.cu)：从一个线程处理一个数，到 `half2`、`float4` 和打包读写。
3. [`kernels/common/pack.cuh`](./kernels/common/pack.cuh)：理解字节数、内存对齐、类型重解释及共用存储。
4. [`kernels/elementwise/test.py`](./kernels/elementwise/test.py)：理解即时编译、预热、基准测量和正确性检查。

**不要直接把原性能测试改成任意奇数尺寸。** 部分向量化实现有对齐及整块大小限制，且包装函数没有完整验证输入。中文注释会指出这些限制，但不会把文档提交伪装成算法修复。

<a id="news"></a>
## 项目动态（原 README 的历史记录）

以下日期和性能数字来自基线版本的原 README，不是中文分支重新测得的结果。

- **2026-05-21**：[Top-K / Top-P 采样](./kernels/sampling/)——发布针对小 K、小批量场景的模板特化与 Split-K 采样核函数，采用寄存器内插入排序和分层合并。原作者表示，相关经验帮助 FlashInfer 实现了 2～4 倍提速；背景见 [FlashInfer #3389](https://github.com/flashinfer-ai/flashinfer/issues/3389)。
- **2026-04-26**：[SM120 Flash Decoding](./kernels/flash_decode/)——使用 TMA、双 K 缓冲区和单 V 缓冲区，提高显存带宽利用率。此处链接按实际目录修正。
- **2026-04-16**：[SM120 Flash Attention](./kernels/flash_attn/)——使用 TMA 降低寄存器压力并改善 L2 缓存局部性；原文报告在其测试的预填充序列长度上超过 `torch.sdpa`（FA2），最好约 1.2 倍。
- **2026-04-04**：[SM120 HGEMM](./kernels/hgemm_sm120/)——FP16/BF16 Tensor Core 实现，组合 TMA 拷贝、三级流水线、双寄存器缓冲、Swizzle、`ldmatrix` 和 `mma`；原文报告超过 cuBLAS。
- **2026-03-18**：[HGEMM](./kernels/hgemm/)——组合 `cp.async`、双共享内存缓冲、Swizzle、`ldmatrix` 和 `mma`；原文报告超过 cuBLAS。
- **2026-03-10**：[TF32 SGEMM](./kernels/sgemm/)——基于 Tensor Core，组合 `cp.async`、双共享内存缓冲、Swizzle、`ldmatrix` 和 `mma`；原文报告超过 cuBLAS。
- **2026-02-27**：[SGEMM](./kernels/sgemm/)——基于 SIMT，组合共享内存、Swizzle、双缓冲和合并读写；原文报告超过 cuBLAS。

## 目录

- [项目动态](#news)
- [环境准备](#prerequisites)
- [核函数目录](#kernels)
  - [入门](#easy)
  - [中级](#medium)
  - [高级](#hard)
- [DSL 核函数](#dsl-kernels)
- [硬件实验](#samples)
- [博客与文章](#blogs)
- [Star 历史](#star-history)
- [参考资料与许可](#references)

<a id="prerequisites"></a>
## 环境准备

原 README 列出的概括性要求是：NVIDIA GPU（计算能力 6.0+）、CUDA Toolkit 11.0+、C++ 编译器（GCC / Clang / MSVC）、CMake 3.18+（可选但推荐）及用于绑定与性能比较的 PyTorch。

**这些不是所有源码共同适用的最低版本保证。** 例如本次读取的 `elementwise.cu` 包含 `cuda_fp8.h`；其测试脚本还导入 FlashInfer，且使用 C++17。新架构指令和 FlashInfer 的具体版本可能要求更新的 GPU、Toolkit、PyTorch 或操作系统。请按模块源码和对应依赖的官方兼容性说明配套安装，不要仅凭上面的概括性版本判断兼容。

原作者推荐用 [NVIDIA PyTorch NGC 容器](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)准备开发环境。容器并不自动保证所有额外依赖和所有架构特化核函数都可用。

先在自己的环境执行：

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`nvidia-smi` 能运行不等于已安装 `nvcc`；安装 CUDA 版 PyTorch 也不等于已安装编译扩展所需的完整 Toolkit。安装与排错的进一步说明见 [入门教程](./docs/zh-CN/入门教程.md)。

<a id="kernels"></a>
## 核函数目录（原项目标称 170+）

原作者说明，除非另有注明，其测试平台为 RTX 5060 Laptop GPU，并与 PyTorch 2.9+ 比较。以下速度表述都属于原作者在其测试条件下的记录，不是本分支在当前环境的测试结论。

下方保留原索引的主要函数标识符；同名的“打包读写”条目沿用原文写法。**实际可调用名称必须以对应源码末尾的 Python 绑定为准**，不能把文档条目直接当作完整 API 清单。

<a id="easy"></a>
### 入门 ⭐️～⭐️⭐️

- [elementwise：逐元素加法](./kernels/elementwise/)
  - [x] `elementwise_add`：FP32 / FP16。
  - [x] `elementwise_add_fp16x2`：FP16，每线程处理 2 个元素。
  - [x] `elementwise_add_fp16x8`：FP16，每线程处理 8 个元素。
  - [x] `elementwise_add_fp16x8_packed`：FP16，8 元素打包读写；这里按已核对的实际绑定名称补全 `_packed`。
  - [x] PyTorch 算子绑定与正确性检查。
  - 中文补注：源码还提供 `elementwise_add_fp32x4`，详见模块文档。
- [sigmoid：S 形激活](./kernels/sigmoid/)
  - [x] `sigmoid`：FP32 / FP16。
  - [x] `sigmoid_fp16x2`：FP16，2 元素向量化。
  - [x] `sigmoid_fp16x8`：FP16，8 元素向量化。
  - [x] `sigmoid_fp16x8`：FP16，8 元素打包读写版本。
  - [x] PyTorch 算子绑定与正确性检查。
- [swish / SiLU：平滑激活](./kernels/silu/)
  - [x] `swish`：FP32 / FP16。
  - [x] `swish_fp16x2`：FP16，2 元素向量化。
  - [x] `swish_fp16x8`：FP16，8 元素向量化。
  - [x] `swish_fp16x8`：FP16，8 元素打包读写版本。
  - [x] PyTorch 算子绑定与正确性检查。
  - 路径补注：原 README 写作 `kernels/swish`，基线目录实际为 `kernels/silu`；上方条目名称保留原索引，不作为已核验的绑定名。
- [relu：线性整流激活](./kernels/relu/)
  - [x] `relu`：FP32 / FP16。
  - [x] `relu_fp16x2`：FP16，2 元素向量化。
  - [x] `relu_fp16x8`：FP16，8 元素向量化。
  - [x] `relu_fp16x8`：FP16，8 元素打包读写版本。
  - [x] PyTorch 算子绑定与正确性检查。
- [relu6：将 ReLU 限制在 0～6](./kernels/relu6/)
  - [x] `relu6`：FP32 / FP16。
  - [x] `relu6_fp16x2`：FP16，2 元素向量化。
  - [x] `relu6_fp16x8`：FP16，8 元素向量化。
  - [x] `relu6_fp16x8`：FP16，8 元素打包读写版本。
  - [x] PyTorch 算子绑定与正确性检查。
- [elu：指数线性单元](./kernels/elu/)
  - [x] `elu`：FP32 / FP16。
  - [x] `elu_fp16x2`：FP16，2 元素向量化。
  - [x] `elu_fp16x8`：FP16，8 元素向量化。
  - [x] `elu_fp16x8`：FP16，8 元素打包读写；原文报告通过 `half2` 内建函数获得约 2 倍提速。
  - [x] PyTorch 算子绑定与正确性检查。
- [gelu：高斯误差线性单元](./kernels/gelu/)
  - [x] `gelu`：FP32 / FP16。
  - [x] `gelu_fp16x2`：FP16，2 元素向量化。
  - [x] `gelu_fp16x8`：FP16，8 元素向量化。
  - [x] `gelu_fp16x8`：FP16，8 元素打包读写版本。
  - [x] PyTorch 算子绑定与正确性检查。
- [hardswish：分段近似的 Swish](./kernels/hardswish/)
  - [x] `hardswish`：FP32 / FP16。
  - [x] `hardswish_fp16x2`：FP16，2 元素向量化。
  - [x] `hardswish_fp16x8`：FP16，8 元素向量化。
  - [x] `hardswish_fp16x8`：FP16，8 元素打包读写版本。
  - [x] PyTorch 算子绑定与正确性检查。
- [embedding：嵌入查表](./kernels/embedding/)
  - [x] `embedding`：FP32 / FP16。
  - [x] `embedding_fp32x4`：FP32，4 元素向量化。
  - [x] `embedding_fp32x4`：FP32，4 元素打包读写版本。
  - [x] `embedding_fp16x2`：FP16，2 元素向量化。
  - [x] `embedding_fp16x8`：FP16，8 元素向量化。
  - [x] `embedding_fp16x8`：FP16，8 元素打包读写版本。
  - [x] PyTorch 算子绑定与正确性检查。
- [rope：旋转位置编码](./kernels/rope/)
  - [x] PyTorch 朴素 RoPE。
  - [x] 使用余弦/正弦查找表的 PyTorch RoPE。
  - [x] `rope_fp32`：原文报告比朴素 PyTorch 实现快约 10 倍。
  - [x] `rope_fp32x4`：FP32 向量化；原文报告在较大规模上快数十倍。
  - [x] PyTorch 算子绑定与正确性检查。

<a id="medium"></a>
### 中级 ⭐️⭐️～⭐️⭐️⭐️

- [reduce：基于 Warp shuffle 的规约](./kernels/reduce/)
  - [x] `reduce_sum`：FP32 / FP16。
  - [x] `reduce_sum_fp16x2`：FP16，2 元素向量化。
  - [x] `reduce_sum_fp16x8_packed`：FP16，8 元素打包读写。
  - [x] `reduce_sum`：INT8。
  - [x] `reduce_sum_i8x16_packed`：INT8，16 元素打包读写。
  - [x] `reduce_sum_i8x16_packed`：INT8，16 元素打包并使用 `dp4a`；原文报告比朴素 PyTorch 快数十倍。
  - [x] `reduce_sum_i8x64_packed`：INT8，64 元素打包并使用 `dp4a`。
  - [x] PyTorch 算子绑定与正确性检查。
- [dot_product：点积](./kernels/dot_product/)
  - [x] `dot_product`：FP32 / FP16。
  - [x] `dot_product_fp32x4`：FP32，4 元素向量化。
  - [x] `dot_product_fp16x2`：FP16，2 元素向量化。
  - [x] `dot_product_fp16x8`：FP16，8 元素打包读写。
  - [x] PyTorch 算子绑定与正确性检查。
- [cumsum：基于 Warp shuffle 的扫描/前缀和](./kernels/cumsum/)
  - [x] 朴素 PyTorch 前缀和。
  - [x] `cumsum_fp32`：FP32。
  - [x] `cumsum_fp32x4`：FP32，4 元素向量化。
  - [x] `cumsum_bf16`：BF16。
  - [x] `cumsum_bf16x8_packed`：BF16，8 元素打包读写。
  - [x] `cumsum_fp32x4_split_k`：Split-K，两遍处理。
  - [x] PyTorch 算子绑定与结果差异检查。
- [normalization：标准化 `(x - mean) / std`](./kernels/normalization/)
  - [x] 朴素 PyTorch 标准化。
  - [x] `norm_fp32`：FP32。
  - [x] `norm_fp32x4`：FP32，4 元素向量化。
  - [x] `norm_fp16`：FP16。
  - [x] `norm_fp16x8_packed`：FP16，8 元素打包读写。
  - [x] `norm_fp32x4_split_k`：Split-K，两遍处理。
  - [x] PyTorch 算子绑定与结果差异检查。
- [softmax：数值稳定的在线 Softmax](./kernels/softmax/)
  - [x] 一遍处理（One-pass）。
    - [x] 数值稳定的在线 Softmax：FP32 / FP16。
    - [x] FP32，4 元素向量化。
    - [x] FP16，8 元素向量化，纯寄存器计算与打包读写。
    - [x] `medium` 版本：FP16，中等寄存器用量与共享内存配合、打包读写。
    - [x] `extreme` 版本：FP16，更多寄存器与共享内存配合、打包读写。
  - [x] 两遍处理（Two-pass），用于超过单线程块处理能力的大行。
    - [x] `arbitrary` 版本：FP16，较多寄存器、共享内存与打包读写。
    - [x] `split-k` 版本：FP16，较多寄存器、共享内存与打包读写。
  - [x] PyTorch 算子绑定与正确性检查。
- [rmsnorm：均方根归一化](./kernels/rmsnorm/)
  - [x] 朴素 PyTorch RMSNorm，作为基准。
  - [x] `rmsnorm`：FP32 / FP16。
  - [x] `rmsnorm_fp32x4`：FP32，4 元素向量化。
  - [x] `rmsnorm_fp32x4_smem`：使用共享内存。
  - [x] `rmsnorm_fp16x8`：FP16，8 元素打包读写。
  - [x] `rmsnorm_fp16x8_smem`：FP16，8 元素打包读写与共享内存。
  - [x] PyTorch 算子绑定与正确性检查。
- [transpose：矩阵转置](./kernels/mat_transpose/)
  - [x] `transpose_coalesced_read`：以输入为中心，实现合并的全局内存读取。
  - [x] `transpose_coalesced_write`：以输出为中心，实现合并的全局内存写入。
  - [x] `transpose_smem`：共享内存分块转置。
  - [x] `transpose_smem_bcf`：避免共享内存 Bank 冲突。
  - [x] `transpose_smem_packed_bcf`：避免 Bank 冲突，采用 `float4` 向量化读写。
  - [x] `transpose_smem_swizzled_packed`：共享内存 Swizzle 重排与 `float4` 向量化读写。
  - [x] PyTorch 算子绑定与正确性检查。

<a id="hard"></a>
### 高级 ⭐️⭐️⭐️～⭐️⭐️⭐️⭐️

- [sgemv：单精度矩阵—向量乘法](./kernels/sgemv/)
  - [x] `gemv_fp32`：基准实现。
  - [x] `gemv_fp32x4`：4 元素向量化读取。
  - [x] PyTorch 算子绑定与正确性检查。
- [sgemm：SIMT 单精度矩阵乘法](./kernels/sgemm/)
  - [x] `sgemm_cublas`：cuBLAS FP32 基准。
  - [x] `sgemm_tiling`：向量化读写、线程块分块和共享内存。
  - [x] `sgemm_at_tiling`：将 A 转置后存入共享内存；原文描述为 4 路 Bank 冲突、内层循环使用 `float4` 加载。
  - [x] `sgemm_at_bcf_swizzling`：A 转置与 Swizzle，避免 Bank 冲突。
  - [x] `sgemm_at_bcf_swizzling_rw`：进一步对 C 的写回进行合并访问。
  - [x] `sgemm_at_bcf_swizzling_dbf_rw`：增加双缓冲流水线；原文报告超过 cuBLAS。
  - [x] PyTorch 算子绑定与正确性检查。
- [sgemm_tf32：TF32 Tensor Core 矩阵乘法](./kernels/sgemm/)
  - [x] `sgemm_cublas_tf32`：cuBLAS TF32 基准。
  - [x] `sgemm_tf32_bt`：A/B 向量化读取、B 转置后存入共享内存、`ldmatrix` 和 `mma`。
  - [x] `sgemm_tf32_bt_swizzle`：进一步避免 A 共享内存访问的 Bank 冲突。
  - [x] `sgemm_tf32_bt_swizzle_dbf`：增加网格重排；原文报告性能为 cuBLAS 的 97%～102%。
  - [x] `sgemm_tf32_swizzle_bcf`：`cp.async`、用 Warp shuffle 转置 B 的寄存器数据、避免 A/B 共享内存冲突及网格重排。
  - [x] `sgemm_tf32_swizzle_bcf_dbf`：增加双缓冲；原文报告超过 cuBLAS。
  - [x] PyTorch 算子绑定与正确性检查。
- [hgemm：FP16 / BF16 矩阵乘法](./kernels/hgemm/)
  - [x] `hgemm_cublas`：cuBLAS BF16/FP16 基准。
  - [x] `hgemm_naive`：BF16/FP16，`ldmatrix` 与 `mma`。
  - [x] `hgemm_bcf`：BF16/FP16、`ldmatrix`、`mma` 与 A/B 共享内存 Swizzle；原文报告为 cuBLAS 的 95%～99%。
  - [x] `hgemm_bcf_dbf`：增加双缓冲；原文报告超过 cuBLAS。
  - [x] `hgemm_bcf_dbf_rw`：进一步合并全局内存读写；原文报告超过 cuBLAS。
  - [x] PyTorch 算子绑定与正确性检查。
- [hgemm_sm120：SM120 架构上的矩阵乘法](./kernels/hgemm_sm120/)
  - [x] `hgemm_cublas`：基准。
  - [x] `hgemm_bcf_dbf_rw`：另一基准，组合 `cp.async`、`ldmatrix`、`mma`、双缓冲和合并全局内存读写。
  - [x] `hgemm_k_stages`：`cp.async`、`ldmatrix`、`mma`、两级/三级缓冲和合并全局内存访问。
  - [x] `hgemm_tma_r_k_stages_64`：TMA 读取、`ldmatrix`、`mma`；一个块、三级流水线、128×128×64 分块、双寄存器缓冲。
  - [x] `hgemm_tma_r_k_stages_32`：TMA 读取、`ldmatrix`、`mma`；两个块、三级流水线、128×128×32 分块、双寄存器缓冲。
  - [x] PyTorch 算子绑定与结果差异检查。
- [SM120 Flash Attention](./kernels/flash_attn/)（原索引拼写为 `flash_attantion_sm120`）
  - [x] `fmha`：单缓冲，BM×BN=64×64；原文说明更适合较小序列长度。
  - [x] PyTorch 算子绑定与结果差异检查。
- [SM120 Flash Decoding](./kernels/flash_decode/)
  - [x] `flash_decode_tma_128`。
  - [x] PyTorch 算子绑定与结果差异检查。
- [Top-K / Top-P 采样](./kernels/sampling/)
  - [x] `sampling_topk_topp_batched`：批量采样。
  - [x] `sampling_topk_topp_split_k`：Split-K 采样。
  - [x] PyTorch 算子绑定与结果差异检查。

<a id="dsl-kernels"></a>
## DSL 核函数

DSL 是“领域专用语言”。这部分可以用来比较手写 CUDA 与更高层 GPU 编程方式；不要求初学者同时学习所有工具。

- [vector_add：向量加法](./kernels_dsl/tilelang/vector_add/)
  - [x] TileLang / Triton 核函数。
  - [x] 与 PyTorch / CUDA 算子进行性能和结果比较。
- [softmax](./kernels_dsl/tilelang/softmax/)
  - [x] TileLang 在线 Softmax，朴素两遍处理。
  - [x] 与 PyTorch / CUDA 算子进行性能和结果比较。

<a id="samples"></a>
## 硬件实验

- [deviceQuery](./samples/deviceQuery/)：查询 GPU 硬件属性。
- [bankConflictNCU](./samples/bank_conflict_ncu/)：共享内存 Bank 冲突与性能分析实验。
- [localMemory](./samples/local_memory/)：理解本地内存地址空间及其性能影响。

<a id="blogs"></a>
## 博客与文章（原项目列出的资源）

### 英文文章的中文标题索引

下方标题已翻译，点击仍会打开原作者的英文文章；不代表外部网页已被翻译或修改。

| 类型 | 文章 | 作者 | 原推荐等级 |
| --- | --- | --- | --- |
| CUDA 入门 | [深入理解 DeviceQuery：认识 GPU 硬件](https://www.wingedge777.com/en/article/5860d3d6e3f62297) | @WingEdge777 | ⭐️⭐️ |
| CUDA 入门 | [CUDA 开发者应该熟悉的数字](https://www.wingedge777.com/en/article/548cec5dfba296d5) | @WingEdge777 | ⭐️⭐️ |
| CUDA 入门 | [理解 CUDA “不存在的存储层级”：本地内存](https://www.wingedge777.com/en/article/7f4c2d9a1e6b8c30) | @WingEdge777 | ⭐️⭐️ |
| CUDA 实战 | [矩阵转置：从 Padding 到 XOR Swizzle 的共享内存优化](https://www.wingedge777.com/en/article/49c4e15376366f8d) | @WingEdge777 | ⭐️⭐️⭐️ |
| CUDA 实战 | [SGEMM：用纯 CUDA C++ 深入优化矩阵乘法](https://www.wingedge777.com/en/article/ce4e1621e32a7b08) | @WingEdge777 | ⭐️⭐️⭐️ |
| CUDA 实战 | [数值稳定的在线 Softmax：任意 hidden_size、一遍/两遍处理、权衡与 Split-K](https://www.wingedge777.com/en/article/8b8f983df949c7c0) | @WingEdge777 | ⭐️⭐️⭐️ |
| CUDA 实战 | [TF32 SGEMM：Tensor Core、cp.async、ldmatrix 与 mma](https://www.wingedge777.com/en/article/ba9e9d9171004edc) | @WingEdge777 | ⭐️⭐️⭐️ |
| CUDA 实战 | [HGEMM：Tensor Core、cp.async、ldmatrix 与 mma](https://www.wingedge777.com/en/article/5a219c62549f9573) | @WingEdge777 | ⭐️⭐️⭐️ |
| CUDA 实战 | [SM120 HGEMM：在 100KB 共享内存中精细优化](https://www.wingedge777.com/en/article/b2ab376d19f52ff4) | @WingEdge777 | ⭐️⭐️⭐️⭐️ |
| CUDA 实战 | [SM120 FMHA：与 torch.sdpa（FlashAttention-2）比较](https://www.wingedge777.com/en/article/7d1151d07a70e2c9) | @WingEdge777 | ⭐️⭐️⭐️⭐️ |
| CUDA 实战 | [SM120 Flash Decoding：与 FlashInfer 单请求解码比较](https://www.wingedge777.com/en/article/7d1151d07a70e2c9) | @WingEdge777 | ⭐️⭐️⭐️⭐️ |

最后两个英文链接沿用原 README，原文中指向同一地址；这里不猜测替代链接。

### 原项目已有的中文博客

| 类型 | 文章 | 作者 | 原推荐等级 |
| --- | --- | --- | --- |
| CUDA 入门 | [深度解读 DeviceQuery：理解你的 GPU 硬件属性](https://zhuanlan.zhihu.com/p/2003100150063724182) | @WingEdge777 | ⭐️⭐️ |
| CUDA 入门 | [CUDA 开发者应该熟悉的数](https://zhuanlan.zhihu.com/p/2004291847716111737) | @WingEdge777 | ⭐️⭐️ |
| CUDA 入门 | [L1/TEX/SMEM：再识 Bank conflict](https://zhuanlan.zhihu.com/p/2012890332749145252) | @WingEdge777 | ⭐️⭐️ |
| CUDA 入门 | [认识 CUDA “不存在的存储层级”：Local Memory](https://zhuanlan.zhihu.com/p/2020146042205643425) | @WingEdge777 | ⭐️⭐️ |
| CUDA 实战 | [矩阵转置：从 Padding 到 Swizzle](https://zhuanlan.zhihu.com/p/2005711788013028455) | @WingEdge777 | ⭐️⭐️⭐️ |
| CUDA 实战 | [RoPE：核函数融合、减少访存和启动开销](https://zhuanlan.zhihu.com/p/2011045579652895890) | @WingEdge777 | ⭐️⭐️⭐️ |
| CUDA 实战 | [SGEMM：优化矩阵乘法的 CUDA C++ 实现](https://zhuanlan.zhihu.com/p/2012477320007533911) | @WingEdge777 | ⭐️⭐️⭐️ |
| CUDA 实战 | [TF32 SGEMM：cp.async、ldmatrix、mma 与 Tensor Core](https://zhuanlan.zhihu.com/p/2016117057889011177) | @WingEdge777 | ⭐️⭐️⭐️ |
| CUDA 实战 | [HGEMM：Tensor Core、cp.async、ldmatrix 与 mma](https://zhuanlan.zhihu.com/p/2017717179840214041) | @WingEdge777 | ⭐️⭐️⭐️ |
| CUDA 实战 | [在线 Softmax：任意 hidden_size、一遍/两遍处理、权衡与 Split-K](https://zhuanlan.zhihu.com/p/2021584505874434002) | @WingEdge777 | ⭐️⭐️⭐️ |
| CUDA 实战 | [SM120 HGEMM：100KB 共享内存中的“微雕”](https://zhuanlan.zhihu.com/p/2023845339765981480) | @WingEdge777 | ⭐️⭐️⭐️⭐️ |
| CUDA 实战 | [SM120 Flash Attention（FMHA）：与 torch.sdpa 比较](https://zhuanlan.zhihu.com/p/2028452875341144849) | @WingEdge777 | ⭐️⭐️⭐️⭐️ |
| CUDA 实战 | [SM120 Flash Decoding（上）：显存带宽优化](https://zhuanlan.zhihu.com/p/2030745157620998667) | @WingEdge777 | ⭐️⭐️⭐️⭐️ |
| CUDA 实战 | [SM120 Flash Decoding（下）：与 FlashInfer 比较](https://zhuanlan.zhihu.com/p/2031825988871525928) | @WingEdge777 | ⭐️⭐️⭐️⭐️ |
| CUDA 实战 | [LLM 推理的最后一公里：Top-K / Top-P、插入排序与块合并](https://zhuanlan.zhihu.com/p/2041177297861039789) | @WingEdge777 | ⭐️⭐️⭐️ |

<a id="star-history"></a>
## Star 历史

[查看上游项目的 Star 历史](https://www.star-history.com/?repos=WingEdge777%2Fvitamin-cuda&type=date&legend=top-left)。原始图表嵌入保留在 [英文 README](./README.en.md#star-history) 中。

<a id="references"></a>
## 参考资料与许可

- [上游 Vitamin-CUDA](https://github.com/WingEdge777/vitamin-cuda)
- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA)
- [NVIDIA CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [PyTorch C++ / CUDA 扩展文档](https://docs.pytorch.org/docs/stable/cpp_extension.html)
- [FlashInfer 安装说明](https://docs.flashinfer.ai/installation.html)

原代码和署名归原权利人所有，许可见 [LICENSE](./LICENSE)。中文说明是学习辅助材料，不替代官方编程规范，也不代表上游作者对本分支新增教学内容的背书。
