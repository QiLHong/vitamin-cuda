# Triton 与 TileLang 向量加法对照

[中文首页](../../../README.md) · [英文原文](./readme.en.md)

原始命令、输出和数值原样保留；本次未复测。请先切换到本目录再执行，以匹配脚本的相对导入路径。

## 功能概览

- [x] Triton 向量加法
- [x] TileLang 向量加法

## 运行测试

```bash
python test.py
```

## 历史输出示例（非本次实测）

## 特殊尺寸测试

```yaml
####################################################################################################
vector add, n: 16777216
torch                                    mean time: 0.321296 ms
triton                                   mean time: 0.303344 ms, speedup: 1.06
tilelang                                 mean time: 0.371168 ms, speedup: 0.87
tilelang_vectorized                      mean time: 0.303312 ms, speedup: 1.06
tilelang_vectorized_eager                mean time: 0.302976 ms, speedup: 1.06
elementwise_add                          mean time: 0.301968 ms, speedup: 1.06
elementwise_add_no_shared                mean time: 0.301440 ms, speedup: 1.07
elementwise_add_fp16x8_packed            mean time: 0.300672 ms, speedup: 1.07
####################################################################################################
vector add, n: 16777215
torch                                    mean time: 0.302304 ms
triton                                   mean time: 0.301488 ms, speedup: 1.00
tilelang                                 mean time: 0.355040 ms, speedup: 0.85
tilelang_vectorized                      mean time: 0.302224 ms, speedup: 1.00
tilelang_vectorized_eager                mean time: 0.304768 ms, speedup: 0.99
elementwise_add_no_shared                mean time: 0.314512 ms, speedup: 0.96
elementwise_add_fp16x8_packed            mean time: 0.301984 ms, speedup: 1.00
####################################################################################################
vector add, n: 16777217
torch                                    mean time: 0.302944 ms
triton                                   mean time: 0.302256 ms, speedup: 1.00
tilelang                                 mean time: 0.355312 ms, speedup: 0.85
tilelang_vectorized                      mean time: 0.302304 ms, speedup: 1.00
tilelang_vectorized_eager                mean time: 0.301728 ms, speedup: 1.00
elementwise_add_no_shared                mean time: 0.314000 ms, speedup: 0.96
elementwise_add_fp16x8_packed            mean time: 0.302336 ms, speedup: 1.00
```

### 完整基准测试的历史记录

```yaml
####################################################################################################
vector add, n: 1024
torch                                    mean time: 0.007360 ms
triton                                   mean time: 0.007328 ms, speedup: 1.00
tilelang                                 mean time: 0.009632 ms, speedup: 0.76
elementwise_add_fp32x4                   mean time: 0.009568 ms, speedup: 0.77
torch                                    mean time: 0.007072 ms
triton                                   mean time: 0.006944 ms, speedup: 1.02
tilelang                                 mean time: 0.006992 ms, speedup: 1.01
tilelang_vectorized                      mean time: 0.006976 ms, speedup: 1.01
tilelang_vectorized_eager                mean time: 0.007024 ms, speedup: 1.01
elementwise_add_fp16x8_packed            mean time: 0.007120 ms, speedup: 0.99
####################################################################################################
vector add, n: 4096
torch                                    mean time: 0.007440 ms
triton                                   mean time: 0.007008 ms, speedup: 1.06
tilelang                                 mean time: 0.007040 ms, speedup: 1.06
elementwise_add_fp32x4                   mean time: 0.006944 ms, speedup: 1.07
torch                                    mean time: 0.006944 ms
triton                                   mean time: 0.007040 ms, speedup: 0.99
tilelang                                 mean time: 0.006944 ms, speedup: 1.00
tilelang_vectorized                      mean time: 0.007248 ms, speedup: 0.96
tilelang_vectorized_eager                mean time: 0.006944 ms, speedup: 1.00
elementwise_add_fp16x8_packed            mean time: 0.006944 ms, speedup: 1.00
####################################################################################################
vector add, n: 32768
torch                                    mean time: 0.007584 ms
triton                                   mean time: 0.007520 ms, speedup: 1.01
tilelang                                 mean time: 0.007616 ms, speedup: 1.00
elementwise_add_fp32x4                   mean time: 0.007440 ms, speedup: 1.02
torch                                    mean time: 0.007040 ms
triton                                   mean time: 0.007280 ms, speedup: 0.97
tilelang                                 mean time: 0.007264 ms, speedup: 0.97
tilelang_vectorized                      mean time: 0.007104 ms, speedup: 0.99
tilelang_vectorized_eager                mean time: 0.007408 ms, speedup: 0.95
elementwise_add_fp16x8_packed            mean time: 0.007136 ms, speedup: 0.99
####################################################################################################
vector add, n: 1048576
torch                                    mean time: 0.040304 ms
triton                                   mean time: 0.045808 ms, speedup: 0.88
tilelang                                 mean time: 0.045536 ms, speedup: 0.89
elementwise_add_fp32x4                   mean time: 0.046864 ms, speedup: 0.86
torch                                    mean time: 0.021648 ms
triton                                   mean time: 0.020240 ms, speedup: 1.07
tilelang                                 mean time: 0.034000 ms, speedup: 0.64
tilelang_vectorized                      mean time: 0.021632 ms, speedup: 1.00
tilelang_vectorized_eager                mean time: 0.021744 ms, speedup: 1.00
elementwise_add_fp16x8_packed            mean time: 0.021808 ms, speedup: 0.99
####################################################################################################
vector add, n: 4194304
torch                                    mean time: 0.156368 ms
triton                                   mean time: 0.160048 ms, speedup: 0.98
tilelang                                 mean time: 0.160784 ms, speedup: 0.97
elementwise_add_fp32x4                   mean time: 0.157584 ms, speedup: 0.99
torch                                    mean time: 0.080544 ms
triton                                   mean time: 0.080608 ms, speedup: 1.00
tilelang                                 mean time: 0.106656 ms, speedup: 0.76
tilelang_vectorized                      mean time: 0.077648 ms, speedup: 1.04
tilelang_vectorized_eager                mean time: 0.079216 ms, speedup: 1.02
elementwise_add_fp16x8_packed            mean time: 0.078336 ms, speedup: 1.03
####################################################################################################
vector add, n: 16777216
torch                                    mean time: 0.599504 ms
triton                                   mean time: 0.595184 ms, speedup: 1.01
tilelang                                 mean time: 0.599456 ms, speedup: 1.00
elementwise_add_fp32x4                   mean time: 0.593072 ms, speedup: 1.01
torch                                    mean time: 0.305136 ms
triton                                   mean time: 0.307120 ms, speedup: 0.99
tilelang                                 mean time: 0.358640 ms, speedup: 0.85
tilelang_vectorized                      mean time: 0.305072 ms, speedup: 1.00
tilelang_vectorized_eager                mean time: 0.304336 ms, speedup: 1.00
elementwise_add_fp16x8_packed            mean time: 0.304736 ms, speedup: 1.00
```

## 实现阅读路线

先读带尾部掩码的 Triton `add_kernel`，再读 TileLang 基础版、每线程八元素的显式向量化版，以及自动推断 `N` 的即时调用版。最后比较显式共享内存暂存与直接使用 fragment 两种写法。`elementwise_add_no_shared` 中变量仍叫 `A_shared` / `B_shared`，但实际分配的是 fragment，应以 `T.alloc_fragment` 为准。

即时调用版按 dtype 位宽计算 `vec`，完整组判断却固定为 `base+7<N`；当前演示使用 FP16，不能据 `dtype` 参数推断任意位宽都已验证。对越过末尾的线程，`T.serial(N-base)` 与 `T.copy` 的保护需结合生成代码确认。中文注释没有修改这些逻辑。

默认入口只执行 `test_sp()`，测试整组长度及前后各差一个元素的长度；`test_all()` 的调用仍被注释。运行需要 PyTorch、Triton、TileLang、NumPy、FlashInfer 和本地 CUDA 扩展构建环境，导入 `kernels.elementwise.test` 会编译 CUDA 加法扩展。各实现并非每一次计时后都立即进行独立误差检查，应按实际 `diff_check` 调用解读输出。计时取中位数，原日志标签 `mean time` 保留不变。
