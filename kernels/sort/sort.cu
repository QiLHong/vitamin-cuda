/*
 * 中文导读：本文件提供 Python 绑定及尚未实现的自定义排序入口。
 * sort_kernel 当前函数体为空，调用 sort 不会生成排序结果；实际 CUB 排序实现在 cub_sort.cu。
 * 保留原有未完成状态和启动代码，不把注释翻译当成算法实现或正确性验证。
 */
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "../common/pack.cuh"

// 待实现的排序核函数占位；当前不读取输入，也不写出输出。
template <const int BLOCK_SIZE = 256, typename T>
__global__ void sort_kernel(T *x, T *y, int n) {}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

// 包装使用 N/block_size 向下取整生成网格，未处理不足一块或末尾部分块。
// 除 Half 外全部走 BF16 指针路径，未完整限制 dtype；即使启动成功，空核函数也无排序结果。
#define binding_func_gen(name, block_size)                                                                             \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int N = a.size(0);                                                                                       \
        const dim3 blocks_per_grid(N / block_size);                                                                    \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        if (a.dtype() == torch::kHalf) {                                                                               \
            name##_kernel<block_size><<<blocks_per_grid, block_size, 0, stream>>>(                                     \
                reinterpret_cast<__half *>(a.data_ptr()), reinterpret_cast<__half *>(b.data_ptr()), N);                \
        } else {                                                                                                       \
            name##_kernel<block_size><<<blocks_per_grid, block_size, 0, stream>>>(                                     \
                reinterpret_cast<__nv_bfloat16 *>(a.data_ptr()), reinterpret_cast<__nv_bfloat16 *>(b.data_ptr()), N);  \
        }                                                                                                              \
    }

binding_func_gen(sort, 256);
// 可用的 CUB 包装在另一编译单元中实现。
extern void cub_sort(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(cub_sort);
    torch_pybinding_func(sort);
}
