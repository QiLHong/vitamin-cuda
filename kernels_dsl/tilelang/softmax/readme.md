# TileLang Softmax

[中文首页](../../../README.md) · [英文原文](./readme.en.md)

此页保留原始命令和性能输出。本次只翻译与补注释，没有执行 GPU 测试。运行前请切换到本目录；脚本通过相对路径导入手写 CUDA Softmax 测试模块，导入过程会编译 CUDA 扩展。

## 功能概览

- [x] TileLang Softmax（基础的两遍扫描版本）

## 运行测试

```bash
python test.py
```

## 历史输出示例（非本次实测）

### 小尺寸行测试：`test_small`

```yaml
#################################################################################################### test_small
bs: 256, hidden_size: 1024
torch                          mean time: 0.015036 ms
softmax_fp32x4                 mean time: 0.010668 ms, speedup: 1.41
tl.float32                     mean time: 0.014306 ms, speedup: 1.05
torch                          mean time: 0.010243 ms
softmax_arbitrary              mean time: 0.009482 ms, speedup: 1.08
tl.float16                     mean time: 0.013906 ms, speedup: 0.74
#################################################################################################### test_small
bs: 256, hidden_size: 4096
torch                          mean time: 0.022326 ms
softmax_fp32x4                 mean time: 0.014161 ms, speedup: 1.58
tl.float32                     mean time: 0.016442 ms, speedup: 1.36
torch                          mean time: 0.021492 ms
softmax_arbitrary              mean time: 0.013525 ms, speedup: 1.59
tl.float16                     mean time: 0.016874 ms, speedup: 1.27
#################################################################################################### test_small
bs: 256, hidden_size: 8192
torch                          mean time: 0.032701 ms
softmax_fp32x4                 mean time: 0.017227 ms, speedup: 1.90
tl.float32                     mean time: 0.023250 ms, speedup: 1.41
torch                          mean time: 0.031312 ms
softmax_arbitrary              mean time: 0.014887 ms, speedup: 2.10
tl.float16                     mean time: 0.016953 ms, speedup: 1.85
#################################################################################################### test_small
bs: 1024, hidden_size: 1024
torch                          mean time: 0.011847 ms
softmax_fp32x4                 mean time: 0.014726 ms, speedup: 0.80
tl.float32                     mean time: 0.031727 ms, speedup: 0.37
torch                          mean time: 0.015049 ms
softmax_arbitrary              mean time: 0.020670 ms, speedup: 0.73
tl.float16                     mean time: 0.029660 ms, speedup: 0.51
#################################################################################################### test_small
bs: 1024, hidden_size: 4096
torch                          mean time: 0.075234 ms
softmax_fp32x4                 mean time: 0.025474 ms, speedup: 2.95
tl.float32                     mean time: 0.037084 ms, speedup: 2.03
torch                          mean time: 0.076734 ms
softmax_arbitrary              mean time: 0.020037 ms, speedup: 3.83
tl.float16                     mean time: 0.027617 ms, speedup: 2.78
#################################################################################################### test_small
bs: 1024, hidden_size: 8192
torch                          mean time: 0.215366 ms
softmax_fp32x4                 mean time: 0.214736 ms, speedup: 1.00
tl.float32                     mean time: 0.204687 ms, speedup: 1.05
torch                          mean time: 0.132761 ms
softmax_arbitrary              mean time: 0.031596 ms, speedup: 4.20
tl.float16                     mean time: 0.052698 ms, speedup: 2.52
```

### 大尺寸行测试：`test_large`

```yaml
#################################################################################################### test_large
bs: 4, hidden_size: 16384
torch                          mean time: 0.012202 ms
softmax_medium                 mean time: 0.009296 ms, speedup: 1.31
softmax_extreme                mean time: 0.014495 ms, speedup: 0.84
softmax_arbitrary              mean time: 0.012852 ms, speedup: 0.95
softmax_splitk                 mean time: 0.017160 ms, speedup: 0.71
tl.float16                     mean time: 0.023353 ms, speedup: 0.52
#################################################################################################### test_large
bs: 4, hidden_size: 32768
torch                          mean time: 0.026349 ms
softmax_medium                 mean time: 0.010866 ms, speedup: 2.42
softmax_extreme                mean time: 0.012803 ms, speedup: 2.06
softmax_arbitrary              mean time: 0.009110 ms, speedup: 2.89
softmax_splitk                 mean time: 0.019735 ms, speedup: 1.34
tl.float16                     mean time: 0.014711 ms, speedup: 1.79
#################################################################################################### test_large
bs: 4, hidden_size: 65536
torch                          mean time: 0.043598 ms
softmax_extreme                mean time: 0.030344 ms, speedup: 1.44
softmax_arbitrary              mean time: 0.013277 ms, speedup: 3.28
softmax_splitk                 mean time: 0.021590 ms, speedup: 2.02
tl.float16                     mean time: 0.021334 ms, speedup: 2.04
#################################################################################################### test_large
bs: 4, hidden_size: 114688
torch                          mean time: 0.022909 ms
softmax_extreme                mean time: 0.015056 ms, speedup: 1.52
softmax_arbitrary              mean time: 0.025445 ms, speedup: 0.90
softmax_splitk                 mean time: 0.019551 ms, speedup: 1.17
tl.float16                     mean time: 0.033421 ms, speedup: 0.69
#################################################################################################### test_large
bs: 4, hidden_size: 262144
torch                          mean time: 0.042772 ms
softmax_arbitrary              mean time: 0.043620 ms, speedup: 0.98
softmax_splitk                 mean time: 0.021454 ms, speedup: 1.99
tl.float16                     mean time: 0.072456 ms, speedup: 0.59
#################################################################################################### test_large
bs: 4, hidden_size: 1048576
torch                          mean time: 0.156257 ms
softmax_arbitrary              mean time: 0.159236 ms, speedup: 0.98
softmax_splitk                 mean time: 0.026065 ms, speedup: 5.99
tl.float16                     mean time: 0.277618 ms, speedup: 0.56
#################################################################################################### test_large
bs: 4, hidden_size: 8388608
torch                          mean time: 1.724874 ms
softmax_arbitrary              mean time: 2.556293 ms, speedup: 0.67
softmax_splitk                 mean time: 0.598768 ms, speedup: 2.88
tl.float16                     mean time: 2.624293 ms, speedup: 0.66
#################################################################################################### test_large
bs: 4, hidden_size: 33554432
torch                          mean time: 6.767271 ms
softmax_arbitrary              mean time: 10.165219 ms, speedup: 0.67
softmax_splitk                 mean time: 2.582483 ms, speedup: 2.62
tl.float16                     mean time: 10.496631 ms, speedup: 0.64
```

## 中文导读与适用范围

每个线程块负责一行。第一遍按最多 8192 个元素分块，计算块内最大值与指数和，再把不同块的统计量换算到共同的最大值基准；第二遍重新读取输入，用整行的最大值和指数和生成输出。统计量采用 FP32；输入与输出类型由 `dtype` 决定。

原注释写“dim 是 2 的幂”，但 `test_large()` 也包含 114688（8192 的 14 倍）。应以实际测试集合和分块条件理解范围。源码没有显式将尾块无效元素填为负无穷，不能只凭 `T.copy` 或向上取整网格就断定所有非整块长度的 Softmax 正确；需要检查对应 TileLang 版本生成的掩码与归约语义。

脚本需要 PyTorch、TileLang、NumPy、FlashInfer 与 CUDA 构建环境。`test_small()` 和 `test_large()` 都在默认入口中执行，最后几项会分配很大的输入。历史计时来自所导入的 `benchmark`，原日志 `mean time` 对应的实现实际使用中位数。历史数字不代表本次通过了正确性、边界或性能检查。
