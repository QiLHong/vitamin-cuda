# Triton 向量加法

[中文首页](../../../README.md) · [英文原文](./readme.en.md)

## 功能概览

- [x] Triton 向量加法

## 运行测试

```bash
python test.py
```

## 历史输出示例（非本次实测）

```yaml
```

## 中文导读

`add_kernel` 的每个 program 处理一个逻辑块，`tl.arange` 生成块内偏移；两次加载和输出存储共用 `offsets<n` 掩码，保护不足整块的尾部。Python 包装按 `triton.cdiv` 向上取整并分配输出。

从本目录运行原始命令，需要安装匹配 GPU 环境的 PyTorch 与 Triton。原脚本先预热，再在 CUDA 同步边界内用主机时钟测量重复调用，输出的是每次调用平均耗时。与 FlashInfer 的中位数计时方式不同，不能混用统计口径。

原 `benchmark` 的 `out` 非空分支没有给末尾返回的 `o` 赋值；默认测试均不传 `out`，不会走这个分支。这里仅记录原限制，未修复代码。本次未运行 GPU 正确性或性能测试。
