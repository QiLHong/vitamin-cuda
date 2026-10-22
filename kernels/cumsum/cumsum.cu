/*
 * 中文导读：对连续二维输入的每一行计算包含当前位置的前缀和，输出形状与输入相同。
 * 普通版本一块一行，按段累积；split-k 版本先处理各分块，再补上此前分块的总和。
 * 标量访问检查单个元素；x4/x8 仅检查组起点，行长分别需为 4/8 的正整数倍。
 * 128 位读写还要求地址按 16 字节对齐；CUDA/连续性检查不等于完整形状、dtype 和设备检查。
 * 这是文档与注释翻译，不修改算法；multi-CTA 单核扫描入口仍是空实现。
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

// 逐轮从前面 offset 个 lane 取累计值；lane_id 较小时不加入无效的前驱。
// 原循环上界写的是 WARP_SIZE，并没有使用模板参数 warp_size；调用需要完整线程束参与。
template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ float _warp_cum_sum(int lane_id, float val) {
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float other_val = __shfl_up_sync(0xffffffff, val, offset);
        if (lane_id >= offset) {
            val += other_val;
        }
    }
    return val;
}

// 一个线程块（CTA）处理一行，delta 保存此前各段的累计和。
template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 256>
__global__ void cumsum_fp32_kernel(float *a, float *b, int n) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    const int num_warp = BLOCK_SIZE / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ float smem[num_warp];
    __shared__ float block_delta;

    float delta = 0.f;
    int row_offset = row * n;
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        int col = i + tid;
        float val = (col < n) ? a[row_offset + col] : 0.0f;
        // 在线程束内做包含当前位置的扫描。
        val = _warp_cum_sum<32>(lane_id, val);

        // 每个线程束的末尾元素写入总和，再由第一个线程束扫描这些总和。
        if (lane_id == 32 - 1)
            smem[warp_id] = val;
        __syncthreads();

        if (warp_id == 0) {
            float tmp_v = tid < num_warp ? smem[tid] : 0.f;
            tmp_v = _warp_cum_sum<num_warp>(lane_id, tmp_v);
            if (tid < num_warp)
                smem[tid] = tmp_v;
        }
        __syncthreads();

        if (warp_id > 0) {
            val += smem[warp_id - 1];
        }

        val += delta;

        if (col < n) {
            b[row_offset + col] = val;
        }

        if (tid == BLOCK_SIZE - 1) {
            block_delta = val;
        }
        __syncthreads();

        delta = block_delta;
    }
}

// BF16 输入输出，使用 FP32 累加；写回时才舍入为 BF16。
template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 256>
__global__ void cumsum_bf16_kernel(__nv_bfloat16 *a, __nv_bfloat16 *b, int n) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    const int num_warp = BLOCK_SIZE / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ float smem[num_warp];
    __shared__ float block_delta;

    float delta = 0.f;
    int row_offset = row * n;
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        int col = i + tid;
        float val = (col < n) ? __bfloat162float(a[row_offset + col]) : 0.0f;
        val = _warp_cum_sum<32>(lane_id, val);

        if (lane_id == 32 - 1)
            smem[warp_id] = val;
        __syncthreads();

        if (warp_id == 0) {
            float tmp_v = tid < num_warp ? smem[tid] : 0.f;
            tmp_v = _warp_cum_sum<num_warp>(lane_id, tmp_v);
            if (tid < num_warp)
                smem[tid] = tmp_v;
        }
        __syncthreads();

        if (warp_id > 0) {
            val += smem[warp_id - 1];
        }

        val += delta;

        if (col < n) {
            b[row_offset + col] = __float2bfloat16(val);
        }

        if (tid == BLOCK_SIZE - 1) {
            block_delta = val;
        }
        __syncthreads();

        delta = block_delta;
    }
}

// 进入此函数前，四个寄存器已保存线程内前缀和；f[3] 是本线程四元素总和。
// 每轮交换前驱线程的总和，并把它加到当前线程的全部四个前缀上。
template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ void _warp_cum_sum(int lane_id, pack128 &val) {
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float other_val = __shfl_up_sync(0xffffffff, val.f[3], offset);
        if (lane_id >= offset) {
#pragma unroll
            for (int k = 0; k < 4; k++)
                val.f[k] += other_val;
        }
    }
}

// 对每线程 N 个 float 寄存器做线程束扫描，以最后一个元素表示该线程的累计总和。
template <const int N, const int warp_size = WARP_SIZE>
__device__ __forceinline__ void _warp_cum_sum_n(int lane_id, float (&val)[N]) {
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float other_val = __shfl_up_sync(0xffffffff, val[N - 1], offset);
        if (lane_id >= offset) {
#pragma unroll
            for (int k = 0; k < N; k++)
                val[k] += other_val;
        }
    }
}

// 每线程连续读取四个 FP32，先做线程内扫描，再累积此前线程与线程束的总和。
// col<n 仅保护组起点；最后不足四个元素时，float4 访问仍可能越过行尾。
template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 1024>
__global__ void cumsum_fp32x4_kernel(float *a, float *b, int n) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    const int num_warp = BLOCK_SIZE / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ float smem[num_warp];
    __shared__ float block_delta;

    float delta = 0.f;
    int row_offset = row * n;
    for (int i = 0; i < n; i += CHUNK_SIZE) {
        int col = i + tid * 4;
        pack128 f4;
        f4.f4 = (col < n) ? FLOAT4(a[row_offset + col]) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int k = 1; k < 4; k++)
            f4.f[k] += f4.f[k - 1];
        _warp_cum_sum<32>(lane_id, f4);

        if (lane_id == 32 - 1)
            smem[warp_id] = f4.f[3];
        __syncthreads();

        if (warp_id == 0) {
            float tmp_v = tid < num_warp ? smem[tid] : 0.f;
            tmp_v = _warp_cum_sum<num_warp>(lane_id, tmp_v);
            if (tid < num_warp)
                smem[tid] = tmp_v;
        }
        __syncthreads();

        float prefix = delta + (warp_id > 0 ? smem[warp_id - 1] : 0.f);
#pragma unroll
        for (int k = 0; k < 4; k++)
            f4.f[k] += prefix;

        if (col < n) {
            FLOAT4(b[row_offset + col]) = f4.f4;
        }

        if (tid == BLOCK_SIZE - 1) {
            block_delta = f4.f[3];
        }

        __syncthreads();

        delta = block_delta;
    }
}

// BF16x8：128 位打包读写，每线程在 FP32 中扫描八个元素。
template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 2048>
__global__ void cumsum_bf16x8_packed_kernel(__nv_bfloat16 *a, __nv_bfloat16 *b, int n) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    const int num_warp = BLOCK_SIZE / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ float smem[num_warp];
    __shared__ float block_delta;

    float delta = 0.f;
    int row_offset = row * n;
    for (int i = 0; i < n; i += CHUNK_SIZE) {
        int col = i + tid * 8;
        pack128 in;
        in.f4 = (col < n) ? LDST128BITS(a[row_offset + col]) : make_float4(0.f, 0.f, 0.f, 0.f);

        float v[8];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f2 = (col < n) ? __bfloat1622float2(in.bf2[j]) : make_float2(0.f, 0.f);
            v[j * 2] = f2.x;
            v[j * 2 + 1] = f2.y;
        }
#pragma unroll
        for (int k = 1; k < 8; k++)
            v[k] += v[k - 1];
        _warp_cum_sum_n<8, 32>(lane_id, v);

        if (lane_id == 32 - 1)
            smem[warp_id] = v[7];
        __syncthreads();

        if (warp_id == 0) {
            float tmp_v = tid < num_warp ? smem[tid] : 0.f;
            tmp_v = _warp_cum_sum<num_warp>(lane_id, tmp_v);
            if (tid < num_warp)
                smem[tid] = tmp_v;
        }
        __syncthreads();

        float prefix = delta + (warp_id > 0 ? smem[warp_id - 1] : 0.f);
#pragma unroll
        for (int k = 0; k < 8; k++)
            v[k] += prefix;

        if (col < n) {
            pack128 out;
#pragma unroll
            for (int j = 0; j < 4; j++)
                out.bf2[j] = __float22bfloat162_rn(make_float2(v[j * 2], v[j * 2 + 1]));
            LDST128BITS(b[row_offset + col]) = out.f4;
        }

        if (tid == BLOCK_SIZE - 1) {
            block_delta = v[7];
        }

        __syncthreads();

        delta = block_delta;
    }
}

// 用 XOR 交换累加线程束内总和，供 split-k 第二轮合并此前分块的总和。
// 此处同样按 WARP_SIZE 循环；模板实参不会缩短实际参与的线程束。
template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ void _warp_reduce(float &val) {
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
}

// split-k 第一轮：计算分块内包含当前位置的前缀和，并写出该分块的总和。
template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 1024>
__global__ void cumsum_fp32x4_split_k_pass_1_kernel(float *a, float *b, float *ws_sum, int n) {
    int tid = threadIdx.x;
    int chunk_id = blockIdx.x;
    int row = blockIdx.y;

    const int num_warp = BLOCK_SIZE / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ float smem[num_warp];

    int row_offset = row * n;
    int col = chunk_id * CHUNK_SIZE + tid * 4;
    pack128 f4;
    f4.f4 = (col < n) ? FLOAT4(a[row_offset + col]) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int k = 1; k < 4; k++)
        f4.f[k] += f4.f[k - 1];
    _warp_cum_sum<32>(lane_id, f4);

    if (lane_id == 32 - 1)
        smem[warp_id] = f4.f[3];
    __syncthreads();

    if (warp_id == 0) {
        float tmp_v = tid < num_warp ? smem[tid] : 0.f;
        tmp_v = _warp_cum_sum<num_warp>(lane_id, tmp_v);
        if (tid < num_warp)
            smem[tid] = tmp_v;
    }
    __syncthreads();

    float prefix = (warp_id > 0 ? smem[warp_id - 1] : 0.f);
#pragma unroll
    for (int k = 0; k < 4; k++)
        f4.f[k] += prefix;

    if (col < n) {
        FLOAT4(b[row_offset + col]) = f4.f4;
    }

    if (tid == BLOCK_SIZE - 1) {
        ws_sum[row * gridDim.x + chunk_id] = f4.f[3];
    }
}

// split-k 第二轮：求当前分块之前所有分块的总和，再加到本块的局部扫描结果上。
template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 1024>
__global__ void cumsum_fp32x4_split_k_pass_2_kernel(float *b, float *ws_sum, int n) {
    int tid = threadIdx.x;
    int chunk_id = blockIdx.x;
    int row = blockIdx.y;
    int blocks_per_row = gridDim.x;

    const int num_warp = BLOCK_SIZE / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ float smem[num_warp];
    __shared__ float prefix_s;

    float val = 0.f;
    for (int i = tid; i < chunk_id; i += BLOCK_SIZE)
        val += ws_sum[row * blocks_per_row + i];
    _warp_reduce(val);
    if (lane_id == 0)
        smem[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        float x = tid < num_warp ? smem[tid] : 0.f;
        _warp_reduce<num_warp>(x);
        if (tid == 0)
            prefix_s = x;
    }
    __syncthreads();
    float prefix = prefix_s;

    int row_offset = row * n;
    int col = chunk_id * CHUNK_SIZE + tid * 4;
    if (col < n) {
        pack128 f4;
        f4.f4 = FLOAT4(b[row_offset + col]);
#pragma unroll
        for (int k = 0; k < 4; k++)
            f4.f[k] += prefix;
        FLOAT4(b[row_offset + col]) = f4.f4;
    }
}

// 多个线程块在单个核函数中协作扫描：尚待实现（TODO），当前函数体为空。
template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 1024>
__global__ void cumsum_fp32x4_multi_cta_scan_kernel(float *a, float *b, int n) {}

// 包装只检查 CUDA 和连续性。a/b 必须是同设备、同 dtype、同形状的二维张量；调用方负责保证。
#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

// 每块 256 线程，每段处理 256*num 个元素；a.size(0) 决定行数，a.size(1) 决定行长。
// 指针重解释不会转换 dtype；宏本身没有检查输入是否与 ptr_type 匹配。
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

binding_func_gen(cumsum_fp32, 1, float);
binding_func_gen(cumsum_fp32x4, 4, float);
binding_func_gen(cumsum_bf16, 1, __nv_bfloat16);
binding_func_gen(cumsum_bf16x8_packed, 8, __nv_bfloat16);

// 工作区每行每块保存一个总和；两轮在同一流中顺序提交以建立先写后读的依赖。
// 第二轮每块重新汇总此前所有块的总和，分块数越多并不必然越快。
void cumsum_fp32x4_split_k(torch::Tensor a, torch::Tensor b) {
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

    dim3 grid(blocks_per_row, bs);
    cumsum_fp32x4_split_k_pass_1_kernel<threads_per_block, chunk_size>
        <<<grid, threads_per_block, 0, stream>>>(reinterpret_cast<float *>(a.data_ptr()),
                                                 reinterpret_cast<float *>(b.data_ptr()),
                                                 ws_sum.data_ptr<float>(),
                                                 lda);
    cumsum_fp32x4_split_k_pass_2_kernel<threads_per_block, chunk_size><<<grid, threads_per_block, 0, stream>>>(
        reinterpret_cast<float *>(b.data_ptr()), ws_sum.data_ptr<float>(), lda);
}

// 此包装绑定了空核函数；调用后不能期待 b 得到前缀和结果。
void cumsum_fp32x4_multi_cta_scan(torch::Tensor a, torch::Tensor b) {
    CHECK_T(a);
    CHECK_T(b);
    const int bs = a.size(0);
    const int lda = a.size(1);
    constexpr int threads_per_block = 256;
    constexpr int vec = 4;
    constexpr int chunk_size = threads_per_block * vec;
    const int blocks_per_row = (lda + chunk_size - 1) / chunk_size;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(blocks_per_row, bs);
    cumsum_fp32x4_multi_cta_scan_kernel<threads_per_block, chunk_size><<<grid, threads_per_block, 0, stream>>>(
        reinterpret_cast<float *>(a.data_ptr()), reinterpret_cast<float *>(b.data_ptr()), lda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(cumsum_fp32);
    torch_pybinding_func(cumsum_fp32x4);
    torch_pybinding_func(cumsum_fp32x4_split_k);
    torch_pybinding_func(cumsum_fp32x4_multi_cta_scan);
    torch_pybinding_func(cumsum_bf16);
    torch_pybinding_func(cumsum_bf16x8_packed);
}
