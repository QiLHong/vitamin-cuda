/*
 * 中文导读：Sigmoid：计算 1/(1+exp(-x))，a 是输入设备数组，b 是输出设备数组。
 * 先看 _sigmoid 的标量运算，再看各核函数的线程映射，最后看 CPU 包装和 Python 绑定。
 * 各线程独立处理元素；在输入输出不发生有害重叠的约定下，不需要块内共享内存或同步。
 * 输入必须同 CUDA 设备、同 dtype、连续、元素数匹配且输出足够；当前设备也应匹配。
 * 原 CHECK_T 并没有验证所有这些关系。N 存在 int 中，空输入与超大输入不在入门约定内。
 * 这是注释版，不修改原逻辑；下面逐版本指出边界限制，不能只凭“有 if”判断安全。
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

// __device__ 是 GPU 端辅助函数，__forceinline__ 尽量消除小函数调用开销。
// 按公式先计算 exp(-x)，再加 1、求倒数；这是直接形式，并非自动采用分段稳定公式。
// 指数运算的误差/溢出及半精度舍入需要单独测试，不能把数学公式当成无限精度计算。
__device__ __forceinline__ float _sigmoid(float x) { return __frcp_rn(1.0f + expf(-x)); }

// half 路径使用半精度内建运算，多个中间步骤可能发生舍入，与先用 FP32 算完再转 half 不同。
__device__ __forceinline__ half _sigmoid(half x) { return hrcp(__hadd(__float2half(1.0f), hexp(__hneg(x)))); }

// half2 的两个分量各算一个 Sigmoid；kOne 是两个 1，不是标量常数自动广播后的新张量。
__device__ __forceinline__ half2 _sigmoid(half2 x) {
    const half2 kOne = __float2half2_rn(1.0f);
    return h2rcp(__hadd2(kOne, h2exp(__hneg2(x))));
}

// FP32 标量核：一线程一元素，先检查 idx<N，再读写，适合学习基本索引。
// blockIdx.x*blockDim.x 是当前块的起始位置，threadIdx.x 是块内偏移。

__global__ void sigmoid_kernel(float *a, float *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        b[idx] = _sigmoid(a[idx]);
    }
}

// FP32 四元素核：线程 t 负责 [4t,4t+3]，x/y/z/w 是同一线程持有的四个分量。
// 注意原实现先读取 float4，再检查完整数据组；尾部标量分支不能证明前面的读取安全。
// 入门采用 N 为 4*256=1024 的正整数倍，且 a/b 首地址按 16 字节对齐。
__global__ void sigmoid_fp32x4_kernel(float *a, float *b, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    float4 a_val = FLOAT4(a[idx]);
    if (idx + 3 < N) {
        float4 b_val = make_float4(_sigmoid(a_val.x), _sigmoid(a_val.y), _sigmoid(a_val.z), _sigmoid(a_val.w));
        FLOAT4(b[idx]) = b_val;
    } else {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (idx + i < N)
                b[idx + i] = _sigmoid(a[idx + i]);
        }
    }
}

// FP16 标量核：线程分配与 FP32 相同，重载选择的是半精度算术。
__global__ void sigmoid_kernel(half *a, half *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        b[idx] = _sigmoid(a[idx]);
    }
}

// FP16 双元素核：完整一对走 half2，否则原代码尝试走标量路径。
// 重要限制：else 内没有检查 idx<N，多余线程也会执行 b[idx]=...，可能越界。
// 入门采用 N 为 2*256=512 的正整数倍，并满足 half2 的 4 字节对齐。
// 不能把“最后一个元素有 else”误认为“所有尾部线程都有保护”。
__global__ void sigmoid_fp16x2_kernel(half *a, half *b, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx + 1 < N) {
        half2 a_val = HALF2(a[idx]);
        HALF2(b[idx]) = _sigmoid(a_val);
    } else {
        b[idx] = _sigmoid(a[idx]);
    }
}

// FP16 八元素核：每线程先读取四组 half2，再逐组判断写回。
// 所有预读都在 if 前；而 idx<N 只检查两元素组的起点，不能独自保护奇数尾部。
// 入门采用 N 为 8*256=2048 的正整数倍，首地址满足 half2 对齐。
__global__ void sigmoid_fp16x8_kernel(half *a, half *b, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 a_val_0 = HALF2(a[idx]);
    half2 a_val_1 = HALF2(a[idx + 2]);
    half2 a_val_2 = HALF2(a[idx + 4]);
    half2 a_val_3 = HALF2(a[idx + 6]);

    if (idx < N) {
        HALF2(b[idx]) = _sigmoid(a_val_0);
    }
    if (idx + 2 < N) {
        HALF2(b[idx + 2]) = _sigmoid(a_val_1);
    }
    if (idx + 4 < N) {
        HALF2(b[idx + 4]) = _sigmoid(a_val_2);
    }
    if (idx + 6 < N) {
        HALF2(b[idx + 6]) = _sigmoid(a_val_3);
    }
}

// FP16 八元素打包核：与上一个版本不同，它先判断完整组，再进行 128 位读取。
// 完整组通过 pack128.f4 搬运 16 字节，经四组 half2 计算后写回。
// else 中每个标量又检查 idx+i<N，保护已启动线程的尾部访问；#pragma unroll 提示展开循环。
// 但 CPU 侧网格公式先做 N/8 的向下取整，某些尾部可能根本没有对应线程。
// 入门采用 N 为 8 的正整数倍、首地址按 16 字节对齐，避免把核内保护与完整调度混淆。
__global__ void sigmoid_fp16x8_packed_kernel(half *a, half *b, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx + 7 < N) {
        pack128 pack_a, pack_b;
        pack_a.f4 = LDST128BITS(a[idx]);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            pack_b.h2[i] = _sigmoid(pack_a.h2[i]);
        }
        LDST128BITS(b[idx]) = pack_b.f4;
    } else {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            if (idx + i < N)
                b[idx + i] = _sigmoid(a[idx + i]);
        }
    }
}
// 仅用于编译期类型选择的标记，不存放 Tensor 数据。
struct Generic {};

// 这里只检查 CUDA 与连续性，不检查同设备、同类型、同尺寸、地址对齐等条件。
// 本分支保留原英文错误字符串，避免翻译改变程序可观察输出。
#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

/*
 * CPU 包装宏：name 决定入口名，num 是每线程元素数，element_dtype 限定允许的类型。
 * a.numel() 是元素数而非字节数；每块 256 线程，原公式先算 N/num，再向上凑块数。
 * 因此 N=0、N<num 或某些非整组尾部需要额外设计，不能假设本包装已经处理。
 * AT_DISPATCH 根据 b.scalar_type() 选择 scalar_t，再用 conditional_t 映射 at::Half 到 half。
 * if constexpr 在编译期筛选分支；虽然分派宏含 double，这里明确拒绝 double。
 * data_ptr() 取设备地址，reinterpret_cast 不转换 dtype、不分配显存。
 * <<<块数,256,0,stream>>> 的 0 表示不申请动态共享内存；stream 是 PyTorch 当前流。
 * name##_kernel 用 ## 拼接函数名，#name 产生字符串；宏解释放在外面以保护续行反斜杠。
 * 启动通常异步，包装返回并不等于 GPU 完成；错误可能到后续同步时才报告。
 */
#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
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
                        total_elements);                                                                               \
                } else {                                                                                               \
                    TORCH_CHECK(false, #name " does not support " + string(toString(b.scalar_type())));                \
                }                                                                                                      \
            }));                                                                                                       \
    }

// 展开为五个 CPU 入口：标量、FP32x4、FP16x2、FP16x8、FP16x8 打包。
binding_func_gen(sigmoid, 1, Generic);
binding_func_gen(sigmoid_fp32x4, 4, float);
binding_func_gen(sigmoid_fp16x2, 2, half);
binding_func_gen(sigmoid_fp16x8, 8, half);
binding_func_gen(sigmoid_fp16x8_packed, 8, half);

// Python 绑定：load() 提供模块名，下面把 CPU 包装函数注册给 Python 调用。
// 包装接收已分配的输出 b 并就地写入，不能按“返回一个新 Tensor”的接口来使用。

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(sigmoid);
    torch_pybinding_func(sigmoid_fp32x4);
    torch_pybinding_func(sigmoid_fp16x2);
    torch_pybinding_func(sigmoid_fp16x8);
    torch_pybinding_func(sigmoid_fp16x8_packed);
}
