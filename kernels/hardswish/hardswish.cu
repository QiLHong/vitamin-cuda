/*
 * 中文导读：Hardswish 逐元素激活。对有限实数计算 x*min(max(x+3,0),6)/6；截断由 min/max 实现。
 * 阅读顺序：设备端标量/half2 运算 → 每线程 1/2/4/8 元素的核函数 → 主机包装与绑定。
 * 输入输出应为同设备、同 dtype、连续且元素数匹配的 CUDA Tensor，输出容量必须足够。
 * CHECK_T 只验证 CUDA 与连续性，不完整检查上述约定、地址对齐和重叠关系。
 * 本文件仅翻译与补充注释；保留原算法及边界限制，未宣称所有尺寸均可安全运行。
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

// __device__ 函数由 GPU 代码调用；下方重载决定实际的计算精度。
__device__ __forceinline__ float _hardswish(float x) {
    float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6 * (1.0f / 6.0f);
}

__device__ __forceinline__ half _hardswish(half x) {
    half relu6 = __hmin(__hmax(x + __float2half(3.0f), __float2half(0.0f)), __float2half(6.f));
    return x * relu6 * __float2half(1.0f / 6.0f);
}

__device__ __forceinline__ half2 _hardswish(half2 x) {
    half2 relu6 = __hmin2(__hmax2(x + __float2half2_rn(3.0f), __float2half2_rn(0.0f)), __float2half2_rn(6.f));
    return x * relu6 * __float2half2_rn(1.0f / 6.0f);
}

// FP32 标量核：线程 t 处理第 t 个元素，先检查 idx<N 再访问。

__global__ void hardswish_kernel(float *a, float *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        b[idx] = _hardswish(a[idx]);
    }
}

// FP32 x4：每线程处理连续四个元素，float4 是一个线程的四分量数据。
// 原代码在边界判断前就做宽读取；后面的标量尾部写回不能保护这次读取。
// 不额外填充时，入门采用 N 为 4*256=1024 的正整数倍，并满足 16 字节对齐。
__global__ void hardswish_fp32x4_kernel(float *a, float *b, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    float4 a_val = FLOAT4(a[idx]);
    if (idx + 3 < N) {
        float4 b_val = make_float4(_hardswish(a_val.x), _hardswish(a_val.y), _hardswish(a_val.z), _hardswish(a_val.w));
        FLOAT4(b[idx]) = b_val;
    } else {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (idx + i < N)
                b[idx + i] = _hardswish(a[idx + i]);
        }
    }
}

// FP16 标量核：索引方式不变，算术采用对应的 half 重载。
__global__ void hardswish_kernel(half *a, half *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        b[idx] = _hardswish(a[idx]);
    }
}

// FP16 x2：完整双元素组使用 half2，但 else 分支没有检查 idx<N。
// 网格向上取整产生的多余线程也可能进入 else，因此不能认为奇数尾部已安全处理。
// 不额外填充时，入门采用 N 为 2*256=512 的正整数倍，并满足 4 字节对齐。
__global__ void hardswish_fp16x2_kernel(half *a, half *b, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx + 1 < N) {
        half2 a_val = HALF2(a[idx]);
        HALF2(b[idx]) = _hardswish(a_val);
    } else {
        b[idx] = _hardswish(a[idx]);
    }
}

// FP16 x8：每线程预读四组 half2，再逐组检查写回的起点。
// 预读先于所有 if，且起点检查不保护奇数末元素；不填充时采用 N 为 2048 的正整数倍。
__global__ void hardswish_fp16x8_kernel(half *a, half *b, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 a_val_0 = HALF2(a[idx]);
    half2 a_val_1 = HALF2(a[idx + 2]);
    half2 a_val_2 = HALF2(a[idx + 4]);
    half2 a_val_3 = HALF2(a[idx + 6]);

    if (idx < N) {
        HALF2(b[idx]) = _hardswish(a_val_0);
    }
    if (idx + 2 < N) {
        HALF2(b[idx + 2]) = _hardswish(a_val_1);
    }
    if (idx + 4 < N) {
        HALF2(b[idx + 4]) = _hardswish(a_val_2);
    }
    if (idx + 6 < N) {
        HALF2(b[idx + 6]) = _hardswish(a_val_3);
    }
}

// FP16 x8 打包读写：完整组先检查 idx+7<N，再以 128 位搬运八个 half。
// 尾部逐元素检查；但包装先计算 N/8，非整组末尾可能根本没有对应线程。
// 入门采用正的 8 倍数及 16 字节对齐；核内保护与网格完整覆盖是两个条件。
__global__ void hardswish_fp16x8_packed_kernel(half *a, half *b, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx + 7 < N) {
        pack128 pack_a, pack_b;
        pack_a.f4 = LDST128BITS(a[idx]);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            pack_b.h2[i] = _hardswish(pack_a.h2[i]);
        }
        LDST128BITS(b[idx]) = pack_b.f4;
    } else {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            if (idx + i < N)
                b[idx + i] = _hardswish(a[idx + i]);
        }
    }
}

// Generic 是编译期类型选择标记，不是用于存储输入的数据结构。
struct Generic {};

// 错误字符串保持原文，便于与上游和已有报错对应。
#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

/*
 * 主机包装：每块 256 线程；num 是每线程元素数，a.numel() 是元素数而非字节数。
 * 网格公式先做 total_elements/num 的整数除法，不能覆盖所有非整组尾部。
 * N 存入 int；空输入、大尺寸索引溢出或网格加法溢出不在这里的入门约定内。
 * AT_DISPATCH 按输出 dtype 分派，if constexpr 排除 double 及不匹配的专用类型。
 * reinterpret_cast 仅改变指针视图，不转换 Tensor dtype；包装没有验证输入输出 dtype 一致。
 * 使用 PyTorch 当前 CUDA 流；核函数启动为异步操作，返回主机不代表设备执行完毕。
 * ## 拼接核函数名，# 将名称转为字符串；解释放在宏外，保持续行反斜杠不变。
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

binding_func_gen(hardswish, 1, Generic);
binding_func_gen(hardswish_fp32x4, 4, float);
binding_func_gen(hardswish_fp16x2, 2, half);
binding_func_gen(hardswish_fp16x8, 8, half);
binding_func_gen(hardswish_fp16x8_packed, 8, half);

// 将 C++ 入口绑定为 Python 模块方法，名称与测试脚本保持一致。

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(hardswish);
    torch_pybinding_func(hardswish_fp32x4);
    torch_pybinding_func(hardswish_fp16x2);
    torch_pybinding_func(hardswish_fp16x8);
    torch_pybinding_func(hardswish_fp16x8_packed);
}
