/*
 * 中文导读：对连续二维张量逐行计算 (x-mean)/std，使用总体方差 E[x²]-E[x]²。
 * 原公式没有 epsilon，也未截断可能因舍入得到的负方差；常量行和数值消减需单独验证。
 * 标量路径逐元素检查；FP32x4/FP16x8 只检查整组起点，分别要求行长为 4/8 的正整数倍。
 * 打包地址需按 16 字节对齐；包装没有完整验证 dtype、形状、输出容量及设备一致性。
 * 这里只增加中文说明，没有改变归一化公式、启动方式或数值行为。
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

// 同一轮交换同时规约一阶和与平方和，避免分别遍历输入。
// 原循环使用 WARP_SIZE 而不是模板参数 warp_size；参与的完整线程束需保持一致。
template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ void _warp_reduce(float &val, float &square) {
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
        square += __shfl_xor_sync(0xffffffff, square, offset);
    }
}

// 一个线程块（CTA）处理一行，先计算整行统计量，再重新读入并归一化。
template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 256>
__global__ void norm_fp32_kernel(float *a, float *b, int n) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    const int num_warp = BLOCK_SIZE / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ float s1[num_warp], s2[num_warp];
    __shared__ float mean_s, std_s;

    int row_offset = row * n;
    float val = 0.f, square = 0.f;
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        int col = i + tid;
        float tmp = (col < n) ? a[row_offset + col] : 0.0f;
        val += tmp;
        square += tmp * tmp;
    }
    // 先在线程束内汇总，再通过共享内存把各线程束结果交给第一线程束。
    _warp_reduce(val, square);
    if (lane_id == 0) {
        s1[warp_id] = val;
        s2[warp_id] = square;
    }
    __syncthreads();
    if (warp_id == 0) {
        float x = tid < num_warp ? s1[tid] : 0.f;
        float sq = tid < num_warp ? s2[tid] : 0.f;
        _warp_reduce<num_warp>(x, sq);
        if (tid == 0) {
            mean_s = x / n;
            // 这里直接开方且没有 epsilon；常量行或舍入消减可能产生特殊值。
            std_s = sqrtf(sq / n - mean_s * mean_s);
        }
    }
    __syncthreads();
    float mean = mean_s;
    float std = std_s;

    for (int i = 0; i < n; i += BLOCK_SIZE) {
        int col = i + tid;
        if (col < n) {
            b[row_offset + col] = (a[row_offset + col] - mean) / std;
        }
    }
}

// 每线程处理四个 FP32；两次遍历分别统计和写出，完整组之外没有标量尾部处理。
template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 1024>
__global__ void norm_fp32x4_kernel(float *a, float *b, int n) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    const int num_warp = BLOCK_SIZE / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ float s1[num_warp], s2[num_warp];
    __shared__ float mean_s, std_s;

    int row_offset = row * n;
    float val = 0.f, square = 0.f;
    for (int i = 0; i < n; i += CHUNK_SIZE) {
        int col = i + tid * 4;
        float4 tmp = (col < n) ? FLOAT4(a[row_offset + col]) : make_float4(0.f, 0.f, 0.f, 0.f);
        val += tmp.x + tmp.y + tmp.z + tmp.w;
        square += tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z + tmp.w * tmp.w;
    }
    _warp_reduce(val, square);
    if (lane_id == 0) {
        s1[warp_id] = val;
        s2[warp_id] = square;
    }
    __syncthreads();
    if (warp_id == 0) {
        float x = tid < num_warp ? s1[tid] : 0.f;
        float sq = tid < num_warp ? s2[tid] : 0.f;
        _warp_reduce<num_warp>(x, sq);
        if (tid == 0) {
            mean_s = x / n;
            std_s = sqrtf(sq / n - mean_s * mean_s);
        }
    }
    __syncthreads();
    float mean = mean_s;
    float std = std_s;

    for (int i = 0; i < n; i += CHUNK_SIZE) {
        int col = i + tid * 4;
        if (col < n) {
            float4 tmp = FLOAT4(a[row_offset + col]);
            float4 out;
            out.x = (tmp.x - mean) / std;
            out.y = (tmp.y - mean) / std;
            out.z = (tmp.z - mean) / std;
            out.w = (tmp.w - mean) / std;
            FLOAT4(b[row_offset + col]) = out;
        }
    }
}

// FP16 输入输出，FP32 中间累积；写回时舍入为 half。
template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 256>
__global__ void norm_fp16_kernel(half *a, half *b, int n) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    const int num_warp = BLOCK_SIZE / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ float s1[num_warp], s2[num_warp];
    __shared__ float mean_s, std_s;

    int row_offset = row * n;
    float val = 0.f, square = 0.f;
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        int col = i + tid;
        float tmp = (col < n) ? __half2float(a[row_offset + col]) : 0.0f;
        val += tmp;
        square += tmp * tmp;
    }
    _warp_reduce(val, square);
    if (lane_id == 0) {
        s1[warp_id] = val;
        s2[warp_id] = square;
    }
    __syncthreads();
    if (warp_id == 0) {
        float x = tid < num_warp ? s1[tid] : 0.f;
        float sq = tid < num_warp ? s2[tid] : 0.f;
        _warp_reduce<num_warp>(x, sq);
        if (tid == 0) {
            mean_s = x / n;
            std_s = sqrtf(sq / n - mean_s * mean_s);
        }
    }
    __syncthreads();
    float mean = mean_s;
    float std = std_s;

    for (int i = 0; i < n; i += BLOCK_SIZE) {
        int col = i + tid;
        if (col < n) {
            float tmp = __half2float(a[row_offset + col]);
            b[row_offset + col] = __float2half((tmp - mean) / std);
        }
    }
}

// FP16x8 打包读写：一次搬运八个 half，转为 FP32 统计和计算，再舍入写出。
template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 2048>
__global__ void norm_fp16x8_packed_kernel(half *a, half *b, int n) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    const int num_warp = BLOCK_SIZE / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ float s1[num_warp], s2[num_warp];
    __shared__ float mean_s, std_s;

    int row_offset = row * n;
    float val = 0.f, square = 0.f;
    for (int i = 0; i < n; i += CHUNK_SIZE) {
        int col = i + tid * 8;
        pack128 tmp;
        tmp.f4 = (col < n) ? LDST128BITS(a[row_offset + col]) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f2 = (col < n) ? __half22float2(tmp.h2[j]) : make_float2(0.f, 0.f);
            val += f2.x + f2.y;
            square += f2.x * f2.x + f2.y * f2.y;
        }
    }
    _warp_reduce(val, square);
    if (lane_id == 0) {
        s1[warp_id] = val;
        s2[warp_id] = square;
    }
    __syncthreads();
    if (warp_id == 0) {
        float x = tid < num_warp ? s1[tid] : 0.f;
        float sq = tid < num_warp ? s2[tid] : 0.f;
        _warp_reduce<num_warp>(x, sq);
        if (tid == 0) {
            mean_s = x / n;
            std_s = sqrtf(sq / n - mean_s * mean_s);
        }
    }
    __syncthreads();
    float mean = mean_s;
    float std = std_s;

    for (int i = 0; i < n; i += CHUNK_SIZE) {
        int col = i + tid * 8;
        if (col < n) {
            pack128 tmp, out;
            tmp.f4 = LDST128BITS(a[row_offset + col]);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 f2 = __half22float2(tmp.h2[j]);
                f2.x = (f2.x - mean) / std;
                f2.y = (f2.y - mean) / std;
                out.h2[j] = __float22half2_rn(f2);
            }
            LDST128BITS(b[row_offset + col]) = out.f4;
        }
    }
}

// split-k 第一轮：每块负责行内一个分块，写出部分和与部分平方和。
template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 1024>
__global__ void norm_fp32x4_split_k_pass_1_kernel(float *a, float *ws_sum, float *ws_sq, int n) {
    int tid = threadIdx.x;
    int chunk_id = blockIdx.x;
    int row = blockIdx.y;

    const int num_warp = BLOCK_SIZE / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ float s1[num_warp], s2[num_warp];

    int row_offset = row * n;
    int col = chunk_id * CHUNK_SIZE + tid * 4;
    float4 tmp = (col < n) ? FLOAT4(a[row_offset + col]) : make_float4(0.f, 0.f, 0.f, 0.f);
    float val = tmp.x + tmp.y + tmp.z + tmp.w;
    float square = tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z + tmp.w * tmp.w;

    _warp_reduce(val, square);
    if (lane_id == 0) {
        s1[warp_id] = val;
        s2[warp_id] = square;
    }
    __syncthreads();
    if (warp_id == 0) {
        float x = tid < num_warp ? s1[tid] : 0.f;
        float sq = tid < num_warp ? s2[tid] : 0.f;
        _warp_reduce<num_warp>(x, sq);
        if (tid == 0) {
            ws_sum[row * gridDim.x + chunk_id] = x;
            ws_sq[row * gridDim.x + chunk_id] = sq;
        }
    }
}

// split-k 第二轮：合并整行的部分统计量得到均值和标准差，再归一化当前分块。
template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 1024>
__global__ void norm_fp32x4_split_k_pass_2_kernel(float *a, float *b, float *ws_sum, float *ws_sq, int n) {
    int tid = threadIdx.x;
    int chunk_id = blockIdx.x;
    int row = blockIdx.y;
    int blocks_per_row = gridDim.x;

    const int num_warp = BLOCK_SIZE / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ float s1[num_warp], s2[num_warp];
    __shared__ float mean_s, std_s;

    float val = 0.f, square = 0.f;
    for (int i = tid; i < blocks_per_row; i += BLOCK_SIZE) {
        val += ws_sum[row * blocks_per_row + i];
        square += ws_sq[row * blocks_per_row + i];
    }
    _warp_reduce(val, square);
    if (lane_id == 0) {
        s1[warp_id] = val;
        s2[warp_id] = square;
    }
    __syncthreads();
    if (warp_id == 0) {
        float x = tid < num_warp ? s1[tid] : 0.f;
        float sq = tid < num_warp ? s2[tid] : 0.f;
        _warp_reduce<num_warp>(x, sq);
        if (tid == 0) {
            mean_s = x / n;
            std_s = sqrtf(sq / n - mean_s * mean_s);
        }
    }
    __syncthreads();
    float mean = mean_s;
    float std = std_s;

    int row_offset = row * n;
    int col = chunk_id * CHUNK_SIZE + tid * 4;
    if (col < n) {
        float4 tmp = FLOAT4(a[row_offset + col]);
        float4 out;
        out.x = (tmp.x - mean) / std;
        out.y = (tmp.y - mean) / std;
        out.z = (tmp.z - mean) / std;
        out.w = (tmp.w - mean) / std;
        FLOAT4(b[row_offset + col]) = out;
    }
}

// a 和 b 仅检查 CUDA、连续性；调用方仍需保证二维同形状、同 dtype、同设备。
#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

// 单行版本每块 256 线程，一次覆盖 256*num 个元素；ptr_type 只改变指针视图。
// 没有依据 a.scalar_type() 做分派，不能向 FP32 入口传 FP16 张量。
#define binding_func_gen(name, num, ptr_type)                                                                          \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int bs = a.size(0);                                                                                      \
        const int lda = a.size(1);                                                                                     \
        const int threads_per_block = 256;                                                                             \
        const int chunk_size = threads_per_block * num;                                                                \
        const int grid = bs;                                                                                           \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<threads_per_block, chunk_size><<<grid, threads_per_block, 0, stream>>>(                          \
            reinterpret_cast<ptr_type *>(a.data_ptr()), reinterpret_cast<ptr_type *>(b.data_ptr()), lda);              \
    }

binding_func_gen(norm_fp32, 1, float);
binding_func_gen(norm_fp32x4, 4, float);
binding_func_gen(norm_fp16, 1, half);
binding_func_gen(norm_fp16x8_packed, 8, half);

// 两个 FP32 工作区保存每行各块的一阶和、平方和；同一 CUDA 流保证两轮先后依赖。
// 第二轮的每个块都合并整行统计量，再写自己的输出分块。
void norm_fp32x4_split_k(torch::Tensor a, torch::Tensor b) {
    CHECK_T(a);
    CHECK_T(b);
    const int bs = a.size(0);
    const int lda = a.size(1);
    constexpr int threads_per_block = 256;
    constexpr int vec = 4;
    constexpr int chunk_size = threads_per_block * vec;
    const int blocks_per_row = (lda + chunk_size - 1) / chunk_size;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());
    auto ws_sum = torch::empty({bs, blocks_per_row}, options);
    auto ws_sq = torch::empty({bs, blocks_per_row}, options);

    dim3 grid(blocks_per_row, bs);
    norm_fp32x4_split_k_pass_1_kernel<threads_per_block, chunk_size><<<grid, threads_per_block, 0, stream>>>(
        reinterpret_cast<float *>(a.data_ptr()), ws_sum.data_ptr<float>(), ws_sq.data_ptr<float>(), lda);
    norm_fp32x4_split_k_pass_2_kernel<threads_per_block, chunk_size>
        <<<grid, threads_per_block, 0, stream>>>(reinterpret_cast<float *>(a.data_ptr()),
                                                 reinterpret_cast<float *>(b.data_ptr()),
                                                 ws_sum.data_ptr<float>(),
                                                 ws_sq.data_ptr<float>(),
                                                 lda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(norm_fp32);
    torch_pybinding_func(norm_fp32x4);
    torch_pybinding_func(norm_fp32x4_split_k);
    torch_pybinding_func(norm_fp16);
    torch_pybinding_func(norm_fp16x8_packed);
}
