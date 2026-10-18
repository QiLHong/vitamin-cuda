# Elementwise：从零理解逐元素加法

[英文原文与完整历史输出](./readme.en.md) · [CUDA 源码](./elementwise.cu) · [测试脚本](./test.py) · [中文首页](../../README.md)

## 概述（原文中文化）

本模块实现逐元素加法，即 `c[i] = a[i] + b[i]`。输入输出元素个数相同，每个位置可以独立计算。

- [x] `elementwise_add`：FP32 / FP16 标量实现。
- [x] `elementwise_add_fp16x2`：FP16 向量化，每线程两个元素。
- [x] `elementwise_add_fp16x8`：FP16 向量化，每线程八个元素。
- [x] `elementwise_add_fp16x8_packed`：FP16 向量化，八元素打包读写。
- [x] PyTorch 算子绑定与结果差异检查。

**源码补注：** 实际绑定还包含 `elementwise_add_fp32x4`，每线程四个 FP32 元素。它在原测试中使用，但未列入原模块 README 的功能清单。

## 先学会手算

```text
A = [1, 2, 3, 4]
B = [5, 6, 7, 8]
C = [6, 8, 10, 12]
```

GPU 不会自动猜测如何分工，程序员需要通过线程索引把工作分给它。标量版本让线程 0 算 `c[0]`、线程 1 算 `c[1]`，依此类推。

本文前四个元素的例子用于解释数学；**不要用长度 4 调用下述所有向量化版本**。性能实现与数学问题的通用定义之间，还隔着输入尺寸、内存对齐和线程启动约定。

## 建议阅读顺序

首先看 `elementwise_add_kernel(float*, ...)`，理解索引与边界；接着看 `half*` 重载，区分指针类型与精度；再读 `half2`、`float4` 和 `pack128` 版本；最后看 `binding_func_gen` 与 `PYBIND11_MODULE`，理解 Python 如何调用 GPU。

`test.py` 的阅读顺序是 `load()` → `benchmark()` → `diff_check()` → 主程序。原脚本中的一些导入未使用，本次没有顺手清理，以便保持纯注释变更。

## 运行测试

### 环境检查

以下为 Linux / Bash 命令，先在仓库根目录执行：

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import numpy; from flashinfer.testing.utils import bench_gpu_time, get_l2_cache_size; print('测试依赖可导入')"
```

本脚本使用 PyTorch 即时编译，因此还需要与 PyTorch/CUDA 匹配的 C++ 编译器、CUDA Toolkit 和 Ninja 构建工具。FlashInfer 的安装包名为 `flashinfer-python`，其兼容性随版本变化；按 [官方安装说明](https://docs.flashinfer.ai/installation.html)选择版本，不要不加区分地升级现有工作环境。

`elementwise.cu` 包含 `cuda_fp8.h`，所以不要把根目录原文中的“CUDA 11.0+”当作此文件的精确最低版本。头文件、编译器、PyTorch 与设备目标必须相互兼容。使用原生 Windows 时尤其要先确认 FlashInfer 支持情况；当前官方安装说明以 Linux 为前提。

### 原文的执行命令

进入本模块目录后，执行原文命令：

```bash
cd kernels/elementwise
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

`TORCH_CUDA_ARCH_LIST` 用来告诉 PyTorch 扩展编译器应生成哪些计算能力目标。原命令选取 `nvidia-smi` 列出的第一张卡；在多 GPU 或 `CUDA_VISIBLE_DEVICES` 重映射的环境，这可能不等于 PyTorch 当前设备。也可以在确认 CUDA 可用后，按 PyTorch 当前设备查询：

```bash
export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; a,b=torch.cuda.get_device_capability(); print(f"{a}.{b}")')"
python test.py
```

若某版 `nvidia-smi` 不支持 `compute_cap` 查询，不要把空值或报错文字当成有效架构。获取到了计算能力也不表示旧版 `nvcc` 一定支持该编译目标。

首次运行会编译扩展。应先看编译错误，不要把它与核函数性能混为一谈。完整测试有多组矩阵尺寸和多次重复；只想验证加载与结果时，可用下方小规模冒烟步骤。

## 不跑完整性能测试的冒烟步骤

下面命令在**仓库根目录**运行。它加载原测试文件中的 `lib`，但不进入该文件的 `if __name__ == '__main__'` 性能测试主程序。依然需要原测试的全部导入依赖。这是文档提供的运行步骤，**本次没有 NVIDIA GPU/nvcc，未实际执行**。

```bash
python - <<'PY'
from pathlib import Path
import runpy
import torch

if not torch.cuda.is_available():
    raise RuntimeError('没有可用的 CUDA 设备，请先检查环境')

path = Path('kernels/elementwise/test.py').resolve()
if not path.is_file():
    raise FileNotFoundError('请在 vitamin-cuda 仓库根目录执行')
lib = runpy.run_path(str(path), run_name='vitamin_cuda_smoke')['lib']

cases = {
    torch.float32: ['elementwise_add', 'elementwise_add_fp32x4'],
    torch.float16: ['elementwise_add', 'elementwise_add_fp16x2',
                    'elementwise_add_fp16x8', 'elementwise_add_fp16x8_packed'],
}
with torch.no_grad():
    for dtype, names in cases.items():
        # 2048 是 4*256 和 8*256 的公倍数；独立新分配的连续张量用于避免切片偏移。
        a = (torch.arange(2048, device='cuda') % 17).to(dtype).contiguous()
        b = torch.full_like(a, 2)
        expected = a + b
        for name in names:
            out = torch.empty_like(a)
            getattr(lib, name)(a, b, out)
            torch.cuda.synchronize()
            torch.testing.assert_close(out, expected, rtol=0, atol=1e-5)
            print(dtype, name, '结果通过')
PY
```

这个例子覆盖两个 dtype 与五个入口，不覆盖所有合法值、所有尺寸、多卡场景或完整数值边界。通过冒烟测试不等于这些核函数已适合生产使用。

## 从线程编号推导数组索引

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

`blockIdx.x` 是块编号，`blockDim.x` 是每块线程数，`threadIdx.x` 是块内编号，全部从 0 计数。若每块 256 线程，第 2 块中的线程 3 负责下标 `2*256+3=515`。

标量版先执行 `if (idx < N)`，再访问 `a[idx]`、`b[idx]` 与 `c[idx]`。这样，多启动的线程不访问数组。向量化版则让线程 t 负责 `2t..2t+1`、`4t..4t+3` 或 `8t..8t+7`；**组内尾元素和多余线程都需要考虑**。

## 各版本的原实现限制：必须先读

下面条件针对当前原包装（每块 256 线程）、非空输入、足够小而不触及 `int`/启动参数上限的 N，且不依赖张量有效范围以外的存储。

| 入口 | 每线程元素数 | 学习时采用的保守条件 | 为什么 |
| --- | ---: | --- | --- |
| `elementwise_add` | 1 | 正整数 N，FP32/FP16 | 访问前检查 `idx<N`；仍不处理空输入启动 |
| `elementwise_add_fp32x4` | 4 | N 为 1024 的正整数倍，三个首地址均按 16 字节对齐 | 源码在边界判断之前读取 `float4`，需要避免尾组及补齐线程预读 |
| `elementwise_add_fp16x2` | 2 | N 为正偶数，首地址满足 `half2` 的 4 字节对齐 | 检查了组起点，但一次读写两个 half |
| `elementwise_add_fp16x8` | 8 | N 为 2048 的正整数倍，首地址满足 `half2` 对齐 | 四组 `half2` 的读取在判断前发生 |
| `elementwise_add_fp16x8_packed` | 8 | N 为 2048 的正整数倍，首地址按 16 字节对齐 | 128 位读取发生在判断前 |

这些是用于入门的保守约定，不是对所有 C++ 类型别名、设备配置或输入组合的形式化安全证明；最终设备指令还取决于编译器。

还有几项容易遗漏的条件：a/b/c 必须同设备、同 dtype、元素数匹配且连续，c 必须容纳全部结果；当前 CUDA 设备应与张量一致，避免输入输出之间存在会造成跨线程覆盖的重叠视图。原 `CHECK_T` 只验证每个张量是连续 CUDA 张量，并没有完整检查这些关系。包装按 `b.scalar_type()` 选择指针类型，**不会替你转换 a/c 的存储格式**。

原网格公式先做 `total_elements / num` 整数除法，再计算块数。因此它不是对任意尾部都完备的调度器；当 N 很小或不能被组宽整除时，可能出现空启动、遗漏尾部或越界的风险。不要认为“后面有 `if`”就一定安全。

## `half2`、`float4` 与 `pack128` 到底是什么？

`half2` 表示两个半精度数，`__hadd2` 对两组数的对应位置分别相加。一个线程仍是一个线程，只是每次处理的数据更多。

`float4` 是四个 FP32 分量组成的类型，共 16 字节。`FLOAT4(a[idx])` 将地址重解释为这种访问视图，不是把一个 float 的数值复制四次。宽访存需要满足对应对齐和有效范围要求。

`pack128` 是 16 字节对齐的联合体，成员共享一段存储：既可通过 `f4` 搬运 16 字节，也可通过四组 `h2` 计算八个 half。这是项目中的低层类型双关写法，不应当作可无条件移植的标准 C++ 通用模式。FP16 与 BF16 虽然都是 16 位，编码和数值特性并不相同。

向量化不保证成倍加速。它可能减少访存指令等开销，但结果还受显存带宽、寄存器使用、数据量和具体编译结果影响。

## Python 到 CUDA 的调用链

```text
Python: lib.elementwise_add(a, b, c)
  -> PYBIND11_MODULE 注册的 C++ 包装函数
  -> CHECK_T 做部分输入检查
  -> AT_DISPATCH... 根据 dtype 选择编译好的分支
  -> 计算 grid/block，取得 PyTorch 当前 CUDA 流
  -> elementwise_add_kernel<<<...>>>(设备指针, ..., N)
  -> GPU 各线程执行
```

`load()` 使用源文件构建动态扩展后返回模块。包装内的 `data_ptr()` 取得设备地址，`reinterpret_cast` 不搬运数据、不申请显存，也不转换数值。输入和输出显存由调用端的 PyTorch 张量管理。

CPU 启动核函数通常是异步的。使用当前流保证的是该流内的顺序；包装返回并不等于 GPU 已经完成。在跨流程序中还需要显式管理依赖关系，本模块没有演示完整的多流同步方案。

## 怎样看懂测试输出？

原文的完整九组历史输出保存在 [readme.en.md 的 Sample output](./readme.en.md#sample-output)。这是原始记录，不是本次运行产生的日志。

`n`、`m` 是二维张量维度，总元素数是 `n*m`；`ms` 是毫秒。`speedup = PyTorch 基线耗时 / 自定义实现耗时`：大于 1 表示这一轮测量更快，小于 1 表示更慢，不是正确性指标。

**原输出写 `mean time`，实际代码使用 `np.median(times)`，计算的是中位数。** 本次保留英文输出字符串，但在这里说明差异。计时工具还设置 `cold_l2_cache=True`，所以不要拿这些数字与其他缓存策略下的结果直接比较。

正确性检查使用最大绝对误差 `<1e-5`。该阈值是当前加法测试的选择，不能直接套用到所有 FP16 矩阵乘法或规约；运行 Python `-O` 还会禁用原脚本中的 `assert`。进行严肃验证时应记录输入、设备、版本、编译选项、误差与测试范围。

## 初学者练习

先在纸上为 N=257 计算标量版的块数和最后一个有效线程，不要直接拿这个 N 测全部向量化版本。随后说明 `idx<N` 为什么不能保护一次读取的第二个 half；再用 2048 个元素比较五个入口，写下哪些改变的是计算方式，哪些改变的是数据搬运宽度。最后在 `pack.cuh` 中算出各 union 成员的字节数，检查为什么不能把成员大小相加。

## 参考依据

本模块解释基于本分支基线的 `elementwise.cu`、`test.py` 与 `pack.cuh`。通用机制参见 [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/)、[PyTorch 扩展文档](https://docs.pytorch.org/docs/stable/cpp_extension.html) 和 [FlashInfer 安装说明](https://docs.flashinfer.ai/installation.html)。本次仅做文档与注释变更，没有进行 GPU 编译、执行或性能复测。
