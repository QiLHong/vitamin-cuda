# CUDA 设备属性查询：中文导读

[中文首页](../../README.md) · [原有中文文章](../../posts/zh/understanding_your_gpu_with_device_query.md)

本目录用 CUDA Runtime API 查询可见设备的属性。`deviceQuery.cpp` 保留了 NVIDIA CUDA Samples 的来源链接、内置映射表和英文输出标签，本分支翻译了注释，方便把源码与终端输出对应起来。

## 构建与运行

先切换到本目录，安装 CUDA Toolkit 和兼容的 C++ 编译器后，再执行以下步骤。这些是待运行命令，不是本次通过的记录。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/deviceQuery
```

原 `CMakeLists.txt` 的最低版本写为 3.18，在架构变量未设置或为假时尝试回退到 `native`。启用 CUDA 时，该变量也可能已由 `CUDAARCHS` 或编译器默认值初始化，因此实际架构以配置输出为准。若使用 `native`，需要 CMake 3.24 或更高版本。较早版本可按工具链支持情况显式设置整数架构编号；例如 `-DCMAKE_CUDA_ARCHITECTURES=86` 表示 8.6，不能在该 CMake 参数里写成 `8.6`。详见 [CMAKE_CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html) 与 [CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)。本次只补充说明，没有改动构建配置。

原 `run.sh` 中的 `$nproc` 是 shell 变量，不是调用 `nproc` 命令；如果使用这个脚本并希望指定构建并行数，需要先设置该变量。

## 怎样读输出

| 输出或属性 | 含义与限制 |
| --- | --- |
| Driver / Runtime Version | 驱动支持的 CUDA 版本与程序链接的 Runtime 版本，应分开理解 |
| Capability Major/Minor | 计算能力编号，用于判断架构特征，不是显存大小 |
| Global memory | 可报告的设备全局内存总量；不等于进程当前剩余可分配量 |
| Multiprocessors | SM 数量，不能单独决定实际任务速度 |
| CUDA Cores/MP | 本源码按架构查表估算，未知架构会回退到表的末项，可能不准确 |
| Shared memory / Registers | 每块或每 SM 的资源上限，与具体核函数可同时驻留的线程块数有关 |
| Warp / Block / Grid limits | 启动与执行分组的限制，不能把网格上限当作并行驻留容量 |
| Concurrent copy and kernel execution | 报告设备能力；实际是否重叠还取决于流、依赖、内存与具体工作负载 |

原程序没有逐项检查所有 Runtime API 返回状态，也不是 GPU 压力测试或性能基准。需要更深入说明时，阅读上方已有中文文章；本次没有运行 CUDA 编译或设备查询。
