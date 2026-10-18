/*
 * 中文学习目标：计算 c[i] = a[i] + b[i]，看懂从“一个线程一个数”到“一个线程一组数”。
 * 推荐阅读顺序：FP32 标量核 -> FP16 标量核 -> half2 -> float4/pack128 -> 绑定宏。
 * __global__ 函数运行在 GPU，后面的 torch::Tensor 包装函数运行在 CPU。
 * 输入约定：a/b/c 在同一 CUDA 设备上，类型及元素个数一致，连续且输出空间足够。
 * 原包装仅检查 CUDA 与连续性，没有完整验证上述约定；当前设备也应与输入一致。
 * 这是保留原算法的中文注释版，不是安全性修复版。空输入、超大输入、未对齐视图、
 * 任意尾部长度并非所有版本都支持，特别留意下面各向量化核的“原实现限制”。
 */
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "../common/pack.cuh"

// 单精度 FP32 标量版本：每个线程负责一个元素，没有跨线程依赖，故不需要块内同步。
// N 是元素个数，不是字节数；连续矩阵也可以按一维数组处理。
__global__ void elementwise_add_kernel(float *a, float *b, float *c, int N) {
    // blockIdx.x 是块编号，blockDim.x 是每块线程数，threadIdx.x 是块内编号。
    // 例如每块 256 线程，第 2 块的第 3 个线程处理下标 2*256+3=515（编号从 0 开始）。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 最后一块可能多启动一些线程；必须先判断下标，再读取和写入数组。
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// FP32 四元素版本：线程 t 对应元素 4t..4t+3，以 float4 视图进行宽访存。
// 原实现限制：下面两次读取发生在边界判断之前，尾部写回分支不能保护这些读取。
// 现有启动方式向上补齐到 256 线程；未经额外填充时，应使用 N 为 4*256 的正整数倍、
// 且首地址按 16 字节对齐的输入。仅有 N%4==0 仍不足以保护补齐线程的预读。
__global__ void elementwise_add_fp32x4_kernel(float *a, float *b, float *c, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    float4 a_val = FLOAT4(a[idx]);
    float4 b_val = FLOAT4(b[idx]);
    // x/y/z/w 是结构体的四个分量，并不是四个线程；每个线程独立完成这四次加法。
    float4 c_val = make_float4(a_val.x + b_val.x, a_val.y + b_val.y, a_val.z + b_val.z, a_val.w + b_val.w);
    if (idx + 3 < N) {
        FLOAT4(c[idx]) = c_val;
    } else {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (idx + i < N)
                c[idx + i] = a[idx + i] + b[idx + i];
        }
    }
}

// 半精度 FP16 标量版本：C++ 函数重载根据指针类型选择此版本。
// __hadd 对两个 half 相加，结果仍是 half；它不是先转成 double 再计算。
__global__ void elementwise_add_kernel(half *a, half *b, half *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

// FP16 双元素版本：half2 装两个 half，__hadd2 分别计算两对元素的和。
// idx<N 在读取前检查，但一次访问两个元素，因此还要求 N 是正偶数、首地址满足 half2 对齐。
// N 为奇数时，idx<N 并不能保证 idx+1<N；原启动宏还先做整数除法，不能视为通用尾部处理。
__global__ void elementwise_add_fp16x2_kernel(half *a, half *b, half *c, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        half2 a_val = HALF2(a[idx]);
        half2 b_val = HALF2(b[idx]);
        HALF2(c[idx]) = __hadd2(a_val, b_val);
    }
}

// FP16 八元素版本：每线程用四个 half2 处理八个连续元素。
// 原实现限制：八个元素在判断前已读入，写回又以两个元素为单位。
// 未额外填充时，应沿用 N 为 8*256 的正整数倍、地址满足 half2 对齐的输入。
// “向量化”指每线程处理的数据更宽，不是线程数量增加八倍。
__global__ void elementwise_add_fp16x8_kernel(half *a, half *b, half *c, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 a_val_0 = HALF2(a[idx]);
    half2 a_val_1 = HALF2(a[idx + 2]);
    half2 a_val_2 = HALF2(a[idx + 4]);
    half2 a_val_3 = HALF2(a[idx + 6]);

    half2 b_val_0 = HALF2(b[idx]);
    half2 b_val_1 = HALF2(b[idx + 2]);
    half2 b_val_2 = HALF2(b[idx + 4]);
    half2 b_val_3 = HALF2(b[idx + 6]);
    if (idx < N) {
        HALF2(c[idx]) = __hadd2(a_val_0, b_val_0);
    }
    if (idx + 2 < N) {
        HALF2(c[idx + 2]) = __hadd2(a_val_1, b_val_1);
    }
    if (idx + 4 < N) {
        HALF2(c[idx + 4]) = __hadd2(a_val_2, b_val_2);
    }
    if (idx + 6 < N) {
        HALF2(c[idx + 6]) = __hadd2(a_val_3, b_val_3);
    }
}

// FP16 八元素、128 位打包读写版本：和上一版算术相同，区别在数据搬运方式。
// pack_a.f4/pack_b.f4 装载 16 字节，再通过 h2[0..3] 解释为四对 half。
// 原实现限制：读取仍在边界判断之前，尾部标量写回不能修复越界读。
// 未额外填充时，使用 N 为 8*256 的正整数倍，且 a/b/c 地址均满足 16 字节对齐。
__global__ void elementwise_add_fp16x8_packed_kernel(half *a, half *b, half *c, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    // 这些是每个线程自己的临时变量，不是多个线程共享的 __shared__ 内存。
    pack128 pack_a, pack_b, pack_c;
    pack_a.f4 = LDST128BITS(a[idx]);
    pack_b.f4 = LDST128BITS(b[idx]);
#pragma unroll
    // #pragma unroll 提示编译器展开固定次数的循环，减少循环控制开销。
    for (int i = 0; i < 4; i++) {
        pack_c.h2[i] = __hadd2(pack_a.h2[i], pack_b.h2[i]);
    }
    if (idx + 7 < N) {
        LDST128BITS(c[idx]) = pack_c.f4;
    } else {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            if (idx + i < N)
                c[idx + i] = pack_c.h[i];
        }
    }
}
// 空类型只用于编译期选择：Generic 表示同一个包装函数可匹配支持的多种浮点类型。
struct Generic {};

// 注意：这里没有检查 a/b/c 的形状、dtype、设备是否相同，也没有检查向量对齐。
// 英文错误字符串保留原样，避免本次文档工作改变程序输出或接口行为。
#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

/*
 * CPU 侧包装宏逐项说明（放在宏外，避免 // 注释吞掉宏末尾的续行反斜杠）：
 * 1. name 是 Python/C++ 的入口名；name##_kernel 用 ## 拼接出 GPU 核函数名称。
 * 2. num 是每线程处理的元素数，element_dtype 限定该入口允许的类型。
 * 3. a.numel() 得到总元素数，原代码存进 int；空张量和超过 int 范围的张量不在安全约定内。
 * 4. 每块 256 线程。blocks_per_grid 先计算 total_elements/num（整数向下取整），
 *    再对线程块数量向上取整，所以它不是对任意 N 都完善的尾部调度方案。
 * 5. getCurrentCUDAStream() 使用 PyTorch 当前流，让核函数跟随当前流中的执行顺序。
 * 6. AT_DISPATCH... 根据 b.scalar_type() 选择 scalar_t；它不会自动转换 a/c 的真实数据。
 * 7. conditional_t 在编译期选择类型；把 at::Half 映射为 CUDA 的 half。
 * 8. if constexpr 在编译期筛选合法分支。虽然分派宏包含 double，但此处明确拒绝 double。
 * 9. <<<块数, 每块线程数, 动态共享内存字节数, 流>>> 是 CUDA 核启动语法；这里共享内存为 0。
 * 10. data_ptr() 取得设备内存地址；reinterpret_cast 只改变指针视图，不分配内存或转换 dtype。
 * 核启动是异步的；包装返回不表示 GPU 已完成，错误可能在后面的同步操作处才报告。
 */
#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a, torch::Tensor b, torch::Tensor c) {                                                     \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        CHECK_T(c);                                                                                                    \
        const int total_elements = a.numel();                                                                          \
        const int threads_per_block = 256;                                                                             \
        const int blocks_per_grid = (total_elements / num + threads_per_block - 1) / threads_per_block;                \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(                                                                           \
            b.scalar_type(), #name, ([&] {                                                                             \
                using namespace std;                                                                                   \
                using cuda_t = conditional_t<is_same_v<scalar_t, at::Half>, half, scalar_t>;                           \
                using UserT = conditional_t<is_same_v<element_dtype, Generic>, cuda_t, element_dtype>;                 \
                using CastT = conditional_t<is_same_v<UserT, double>, float, UserT>;                                   \
                                                                                                                       \
                constexpr bool is_generic = is_same_v<element_dtype, Generic>;                                         \
                constexpr bool is_match = is_same_v<cuda_t, element_dtype>;                                            \
                constexpr bool is_double = is_same_v<scalar_t, double>;                                                \
                                                                                                                       \
                if constexpr (!is_double && (is_generic || is_match)) {                                                \
                    name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(                                  \
                        reinterpret_cast<CastT *>(a.data_ptr()),                                                       \
                        reinterpret_cast<CastT *>(b.data_ptr()),                                                       \
                        reinterpret_cast<CastT *>(c.data_ptr()),                                                       \
                        total_elements);                                                                               \
                } else {                                                                                               \
                    TORCH_CHECK(false, #name " does not support " + string(toString(b.scalar_type())));                \
                }                                                                                                      \
            }));                                                                                                       \
    }

// 以下五行展开成五个 CPU 包装函数，避免为不同打包宽度重复写同样的启动代码。
binding_func_gen(elementwise_add, 1, Generic);
binding_func_gen(elementwise_add_fp32x4, 4, float);
binding_func_gen(elementwise_add_fp16x2, 2, half);
binding_func_gen(elementwise_add_fp16x8, 8, half);
binding_func_gen(elementwise_add_fp16x8_packed, 8, half);

// Python 绑定入口：load() 编译扩展时提供 TORCH_EXTENSION_NAME。
// 例如 Python 的 lib.elementwise_add(...) 最终进入上面宏生成的包装函数。

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(elementwise_add);
    torch_pybinding_func(elementwise_add_fp32x4);
    torch_pybinding_func(elementwise_add_fp16x2);
    torch_pybinding_func(elementwise_add_fp16x8);
    torch_pybinding_func(elementwise_add_fp16x8_packed);
}
