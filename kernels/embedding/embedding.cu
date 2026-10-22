/*
 * 中文导读：每个线程块根据一个 INT32 索引，从嵌入表复制一行到输出。
 * a 是 [seq_len] 索引，b 是 [词表大小,emd_dim] 表，c 是 [seq_len,emd_dim] 输出。
 * 索引 dtype、合法范围及表/输出形状关系未在包装中完整检查；INT64 地址不能直接按 int* 使用。
 * x4/x8 仅检查组起点，需要整组维度；packed 地址需按 16 字节对齐。
 * 这些函数只搬运数据，不包含梯度计算或高层 embedding 的其他选项。
 */
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "../common/pack.cuh"

// FP32 标量查表：线程沿嵌入维度按 blockDim.x 步长复制。

__global__ void embedding_kernel(int *a, float *b, float *c, int emd_dim) {
    int out_offset = blockIdx.x * emd_dim;
    int in_offset = a[blockIdx.x] * emd_dim;
    int idx = threadIdx.x;
    for (int i = idx; i < emd_dim; i += blockDim.x) {
        c[out_offset + i] = b[in_offset + i];
    }
}

// FP32x4：每线程用展开循环复制四个标量；源码的分组不保证某种具体机器访存指令。
__global__ void embedding_fp32x4_kernel(int *a, float *b, float *c, int emd_dim) {
    int out_offset = blockIdx.x * emd_dim;
    int in_offset = a[blockIdx.x] * emd_dim;
    int idx = threadIdx.x * 4;
    for (int i = idx; i < emd_dim; i += blockDim.x * 4) {
#pragma unroll
        for (int j = 0; j < 4; j++)
            c[out_offset + i + j] = b[in_offset + i + j];
    }
}

// FP32x4 packed：通过 128 位访问复制四个 float；不改变数值位模式。
__global__ void embedding_fp32x4_packed_kernel(int *a, float *b, float *c, int emd_dim) {
    int out_offset = blockIdx.x * emd_dim;
    int in_offset = a[blockIdx.x] * emd_dim;
    int idx = threadIdx.x * 4;
    for (int i = idx; i < emd_dim; i += blockDim.x * 4) {
        LDST128BITS(c[out_offset + i]) = LDST128BITS(b[in_offset + i]);
    }
}

// FP16 标量查表：索引关系与 FP32 相同，只改变表和输出的元素类型。
__global__ void embedding_kernel(int *a, half *b, half *c, int emd_dim) {
    int out_offset = blockIdx.x * emd_dim;
    int in_offset = a[blockIdx.x] * emd_dim;
    int idx = threadIdx.x;
    for (int i = idx; i < emd_dim; i += blockDim.x) {
        c[out_offset + i] = b[in_offset + i];
    }
}

// FP16x8：每线程每轮复制八个标量，内部未逐元素检查行尾。
__global__ void embedding_fp16x8_kernel(int *a, half *b, half *c, int emd_dim) {
    int out_offset = blockIdx.x * emd_dim;
    int in_offset = a[blockIdx.x] * emd_dim;
    int idx = threadIdx.x * 8;
    for (int i = idx; i < emd_dim; i += blockDim.x * 8) {
#pragma unroll
        for (int j = 0; j < 8; j++)
            c[out_offset + i + j] = b[in_offset + i + j];
    }
}

// FP16x8 打包读写：128 位一次搬运八个 half，要求整组维度和 16 字节对齐。
__global__ void embedding_fp16x8_packed_kernel(int *a, half *b, half *c, int emd_dim) {
    int out_offset = blockIdx.x * emd_dim;
    int in_offset = a[blockIdx.x] * emd_dim;
    int idx = threadIdx.x * 8;
    for (int i = idx; i < emd_dim; i += blockDim.x * 8) {
        LDST128BITS(c[out_offset + i]) = LDST128BITS(b[in_offset + i]);
    }
}

struct Generic {};

// 这里只检查 CUDA 与连续性；索引范围、INT32 dtype 及输出容量仍由调用方保证。
#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

// 每块对应一个索引，线程数为 min(512,emd_dim)/num。emd_dim<num 时可能得到零线程。
// 按 b 的 dtype 分派，但未验证 c 的 dtype 相同；a 的地址固定重解释为 int*。
#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a, torch::Tensor b, torch::Tensor c) {                                                     \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        CHECK_T(c);                                                                                                    \
        const int seq_len = a.size(0);                                                                                 \
        const int emd_dim = b.size(1);                                                                                 \
        const int threads_per_block = std::min(512, emd_dim) / num;                                                    \
        const int blocks_per_grid = seq_len;                                                                           \
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
                        reinterpret_cast<int *>(a.data_ptr()),                                                         \
                        reinterpret_cast<CastT *>(b.data_ptr()),                                                       \
                        reinterpret_cast<CastT *>(c.data_ptr()),                                                       \
                        emd_dim);                                                                                      \
                } else {                                                                                               \
                    TORCH_CHECK(false, #name " does not support " + string(toString(b.scalar_type())));                \
                }                                                                                                      \
            }));                                                                                                       \
    }

binding_func_gen(embedding, 1, Generic);
binding_func_gen(embedding_fp32x4, 4, float);
binding_func_gen(embedding_fp32x4_packed, 4, float);
binding_func_gen(embedding_fp16x8, 8, half);
binding_func_gen(embedding_fp16x8_packed, 8, half);

// 将已实现的入口绑定到 Python；源码没有 embedding_fp16x2 入口。

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(embedding);
    torch_pybinding_func(embedding_fp32x4);
    torch_pybinding_func(embedding_fp32x4_packed);
    torch_pybinding_func(embedding_fp16x8);
    torch_pybinding_func(embedding_fp16x8_packed);
}
