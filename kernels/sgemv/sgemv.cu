/*
 * 中文导读：A[out_channels,in_channels] 与向量 b 相乘，结果写入 c[out_channels]。
 * gemv 每块负责一行，gemv_fp32x4 每块负责四行；两者都采用线程束内、块内两级求和。
 * x4 版本的宽加载和四行写出没有通用尾部保护：行数/列数须为 4 的正整数倍，并满足 16 字节对齐。
 * a/b/c 应是同设备、连续、形状匹配的 FP32 数据；包装对 c 没有执行 CHECK_T。
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

// 线程束内用 XOR 配对逐轮相加；满掩码要求参与的线程遵守同步调用约定。
template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ float _warp_shuffle_reduce_sum(float val) {
#pragma unroll
    for (int offset = warp_size >> 1; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// 先得到每个线程束的和，由其 lane 0 写共享内存，再让第一个线程束合并这些部分和。
// 当前启动 128 线程，因此实际实例化 num_warp=4；两道块屏障分别保护发布与读取。
template <const int num_warp = 8>
__device__ __forceinline__ float _block_reduce_sum(float val) {
    val = _warp_shuffle_reduce_sum<WARP_SIZE>(val);

    __shared__ float sdata[num_warp];
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();
    if (warp_id == 0) {
        val = lane_id < num_warp ? sdata[lane_id] : 0.f;
        val = _warp_shuffle_reduce_sum<num_warp>(val);
        if (lane_id == 0) {
            sdata[0] = val;
        }
    }
    __syncthreads();
    return sdata[0];
}

// FP32 矩阵—向量乘法：每个线程块计算一行的点积。
template <const int BLOCK_SIZE = 128>
__global__ void gemv_kernel(float *a, float *b, float *c, int out_channels, int in_channels) {
    int x = threadIdx.x;
    int y = blockIdx.x;

    float sum = 0.0f;
    if (y < out_channels) {
        for (int i = x; i < in_channels; i += blockDim.x) {
            sum += a[y * in_channels + i] * b[i];
        }
        sum = _block_reduce_sum<BLOCK_SIZE / WARP_SIZE>(sum);
        if (x == 0) {
            c[y] = sum;
        }
    }
}

// FP32 四元素向量化版本：每个线程块计算四行，复用同一次加载的 b。
template <const int BLOCK_SIZE = 128>
__global__ void gemv_fp32x4_kernel(float *a, float *b, float *c, int out_channels, int in_channels) {
    const int ROWS_PER_BLOCK = 4;

    int tid = threadIdx.x;
    int row_start = blockIdx.x * ROWS_PER_BLOCK;

    float sum[ROWS_PER_BLOCK] = {0.0f};

    for (int k = tid * 4; k < in_channels; k += blockDim.x * 4) {
        float4 vb = LDST128BITS(b[k]);
#pragma unroll
        for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
            int current_row = row_start + r;
            float4 va = LDST128BITS(a[current_row * in_channels + k]);
            sum[r] += va.x * vb.x;
            sum[r] += va.y * vb.y;
            sum[r] += va.z * vb.z;
            sum[r] += va.w * vb.w;
        }
    }
    pack128 out_pack;
#pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
        float res = _block_reduce_sum<BLOCK_SIZE / WARP_SIZE>(sum[r]);
        if (tid == 0) {
            out_pack.f[r] = res;
        }
    }
    if (tid == 0) {
        LDST128BITS(c[row_start]) = out_pack.f4;
    }
}

// 注意 binding_func_gen 只对 a/b 调用此宏；没有对 c 做同样的设备和连续性检查。
// num 决定每块输出 1 行还是 4 行，网格向上取整不等于 x4 核函数具有行尾保护。
#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a, torch::Tensor b, torch::Tensor c) {                                                     \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int out_channels = a.size(0);                                                                            \
        const int in_channels = a.size(1);                                                                             \
        const int threads_per_block = 128;                                                                             \
        const int blocks_per_grid = (out_channels + num - 1) / num;                                                    \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(                                              \
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), out_channels, in_channels);                 \
    }

binding_func_gen(gemv, 1, float);
binding_func_gen(gemv_fp32x4, 4, float);

// 导出供 Python 调用的绑定。

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(gemv);
    torch_pybinding_func(gemv_fp32x4);
}
