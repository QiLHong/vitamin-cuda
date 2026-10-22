# 基数排序（Radix Sort）

[英文原文](./readme.en.md) · [中文入门教程](../../docs/zh-CN/入门教程.md) · [术语表](../../docs/zh-CN/术语表.md)

> 本页保留原功能清单和历史输出；功能勾选不表示本次已运行验证。

## 概述

基数排序相关实现，包含 CUB 调用和尚未完成的自定义核函数。

- [x] cub_sort — BF16
- [ ] sort — BF16
- [x] PyTorch 算子绑定与结果差异检查

## 中文导读与实现状态

`cub_sort` 调用 CUB 的 `DeviceRadixSort::SortKeys` 对 BF16 键排序。第一次用空临时缓冲查询所需字节数，随后在当前 CUDA 设备上申请字节张量作为工作区，再在当前 CUDA 流上调用排序。输入输出必须连续、均为 BF16 且形状一致；当前包装对空输入直接返回，但没有完整检查张量与当前设备的一致性或输入输出重叠关系。

`sort.cu` 中的 `sort_kernel` 函数体为空，`sort` 仍是未实现入口。`test.py` 虽然调用该入口，却复用了已有输出，且最终差异断言被注释掉。因此脚本正常结束不能证明自定义排序正确。不要将翻译完成、绑定存在或勾选的 CUB 项目理解为自定义核函数已经实现。

当前示例输出代码块本身为空，下面按原样保留。本次没有补造性能或正确性日志。

## 运行测试

需要可用的 NVIDIA GPU、匹配的 CUDA/PyTorch 编译环境，以及脚本导入的 NumPy、FlashInfer 等依赖。在当前模块目录执行下列命令；本次未进行 CUDA 编译、GPU 正确性或性能测试。

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 历史示例输出

以下为原仓库保留的记录（含原英文日志和数值），不是本次翻译产生的运行结果。Python 计时统计实际使用中位数 `np.median`，原日志标签 `mean time` 保留不变。

```bash

```
