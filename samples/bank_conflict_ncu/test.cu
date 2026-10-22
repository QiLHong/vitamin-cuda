/* 中文导读：本文件用于观察共享内存访问指标；翻译不对跨 Warp 冲突原因作新的实测结论。 */
#include "../../kernels/common/pack.cuh"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>

// FP32 x4 搬运实验：每线程读四个 float，只将第一个分量乘 2，再经共享内存写回。
// 块内固定 128 个线程，共享数组 512 个 float；核函数没有 N 尾部保护。
__global__ void load_fp32x4_kernel(float *a, float *b, int n) {
    __shared__ float s[512];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 val = FLOAT4(a[idx * 4]);
    val.x *= 2;
    FLOAT4(s[threadIdx.x * 4]) = val;
    // 块内屏障使写入阶段完成后再读取；性能计数器的含义需结合实际生成的指令判断。
    __syncthreads();

    FLOAT4(b[idx * 4]) = FLOAT4(s[threadIdx.x * 4]);
}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

// 网格采用整数除法 N/4/128；本实验输入必须是正的 512 倍数且满足 float4 对齐。
// 保留宏中的原参数与字符串；CHECK_T 未验证全部 shape/dtype/同设备条件。
#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int N = a.size(0);                                                                                       \
        const int threads_per_block = 128;                                                                             \
        const dim3 blocks_per_grid = N / num / threads_per_block;                                                      \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(a.data_ptr<float>(), b.data_ptr<float>(), N); \
    }

binding_func_gen(load_fp32x4, 4, float);

// 绑定 Python 入口，供 Nsight Compute 采样脚本调用。

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { torch_pybinding_func(load_fp32x4); }
