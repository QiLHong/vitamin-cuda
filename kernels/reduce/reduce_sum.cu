/*
 * 中文导读：对整个输入求和，线程局部累积、线程束规约、共享内存合并后再跨块原子累加。
 * 浮点输入用 FP32 累加，最终转换回输入 dtype；INT8 输入用 INT32 累加并返回 INT32。
 * 向量版本只判断组起点，需整组长度和相应对齐；没有任意尾部处理。
 * INT8 包装直接重解释地址而未验证 dtype；尺寸保存为 int，空输入及超大范围需要另外设计。
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

// 在线程束内用 XOR 交换做求和规约；warp_size 决定需要合并的逻辑组大小。
template <const int warp_size = WARP_SIZE, typename T>
__device__ __forceinline__ T _warp_shuffle_reduce(T x) {
#pragma unroll
    for (int mask = warp_size >> 1; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask);
    }
    return x;
}

// 标量路径逐元素判断 i<N，并先将输入转换为 float 再累加。
// 网格步长循环让固定数量的线程块覆盖更长的输入。
template <const int block_size = 256, typename T>
__global__ void reduce_sum_kernel(T *a, float *b, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        sum += static_cast<float>(a[i]);
    }

    const int num_warp = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    __shared__ float smem[num_warp];

    // 每个线程束先规约，再经共享内存把八个线程束的结果交给第一线程束。
    sum = _warp_shuffle_reduce<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = lane_id < num_warp ? smem[lane_id] : 0.f;
        sum = _warp_shuffle_reduce<num_warp>(sum);
        if (tid == 0) {
            atomicAdd(b, sum);
        }
    }
}

// FP32x4：一次读取四个 float 后相加；N 需为 4 的倍数、地址需按 16 字节对齐。
template <const int block_size = 256>
__global__ void reduce_sum_fp32x4_kernel(float *a, float *b, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    float4 a_val;
    for (int i = idx * 4; i < N; i += gridDim.x * blockDim.x * 4) {
        a_val = FLOAT4(a[i]);
        sum += a_val.x + a_val.y + a_val.z + a_val.w;
    }

    const int num_warp = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    __shared__ float smem[num_warp];

    sum = _warp_shuffle_reduce<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = lane_id < num_warp ? smem[lane_id] : 0.f;
        sum = _warp_shuffle_reduce<num_warp>(sum);
        if (tid == 0) {
            atomicAdd(b, sum);
        }
    }
}

// FP16x2：一次读取 half2，转换为两个 float 后累加；N 需为偶数，地址按 4 字节对齐。
template <const int block_size = 256>
__global__ void reduce_sum_fp16x2_kernel(half *a, float *b, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    for (int i = idx * 2; i < N; i += gridDim.x * blockDim.x * 2) {
        float2 p = __half22float2(HALF2(a[i]));
        sum += p.x + p.y;
    }

    const int num_warp = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    __shared__ float smem[num_warp];

    sum = _warp_shuffle_reduce<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = lane_id < num_warp ? smem[lane_id] : 0.f;
        sum = _warp_shuffle_reduce<num_warp>(sum);
        if (tid == 0) {
            atomicAdd(b, sum);
        }
    }
}

// FP16x8：用 128 位访问加载八个 half，逐对转为 FP32 后累加。
// N 需为 8 的倍数，地址按 16 字节对齐；末尾不足八个元素时没有保护。
template <const int block_size = 256>
__global__ void reduce_sum_fp16x8_packed_kernel(half *a, float *b, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    pack128 pack_p;
    for (int i = idx * 8; i < N; i += gridDim.x * blockDim.x * 8) {
        pack_p.f4 = LDST128BITS(a[i]);
#pragma unroll
        for (int t = 0; t < 4; t++) {
            float2 f = __half22float2(pack_p.h2[t]);
            sum += f.x + f.y;
        }
    }

    const int num_warp = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    __shared__ float smem[num_warp];

    sum = _warp_shuffle_reduce<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = lane_id < num_warp ? smem[lane_id] : 0.f;
        sum = _warp_shuffle_reduce<num_warp>(sum);
        if (tid == 0) {
            atomicAdd(b, sum);
        }
    }
}

// INT8 标量输入扩展为 INT32 再相加；总和仍受 INT32 可表示范围限制。
template <const int block_size = 256>
__global__ void reduce_sum_i8_kernel(int8_t *a, int32_t *b, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int32_t sum = 0;
    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        sum += static_cast<int32_t>(a[i]);
    }

    const int num_warp = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    __shared__ int smem[num_warp];

    sum = _warp_shuffle_reduce<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = lane_id < num_warp ? smem[lane_id] : 0;
        sum = _warp_shuffle_reduce<num_warp>(sum);
        if (tid == 0) {
            atomicAdd(b, sum);
        }
    }
}

// 每次打包加载 16 个 INT8，再逐字节转 INT32 累加；长度需为 16 的倍数。
template <const int block_size = 256>
__global__ void reduce_sum_i8x16_packed_kernel(int8_t *a, int32_t *b, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int32_t sum = 0;
    pack128 pack_i8;
    for (int i = idx * 16; i < N; i += gridDim.x * blockDim.x * 16) {
        pack_i8.f4 = LDST128BITS(a[i]);
#pragma unroll
        for (int t = 0; t < 16; t++) {
            sum += static_cast<int32_t>(pack_i8.i8[t]);
        }
    }

    const int num_warp = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    __shared__ int32_t smem[num_warp];

    sum = _warp_shuffle_reduce<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = lane_id < num_warp ? smem[lane_id] : 0;
        sum = _warp_shuffle_reduce<num_warp>(sum);
        if (tid == 0) {
            atomicAdd(b, sum);
        }
    }
}

// beta 的四个字节都是 1；每次 __dp4a 累加四个有符号 INT8 与 1 的乘积。
// 四次指令合并一组 16 字节；这是打包字节点积，不是把整个 int 当普通数相乘。
template <const int block_size = 256>
__global__ void reduce_sum_i8x16_packed_dp4a_kernel(int8_t *a, int32_t *b, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int32_t sum = 0;
    const int beta = 0x01010101;
    int4 value;
    for (int i = idx * 16; i < N; i += gridDim.x * blockDim.x * 16) {
        LDST128BITS(value) = LDST128BITS(a[i]);
        sum = __dp4a(value.x, beta, sum);
        sum = __dp4a(value.y, beta, sum);
        sum = __dp4a(value.z, beta, sum);
        sum = __dp4a(value.w, beta, sum);
    }

    const int num_warp = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    __shared__ int32_t smem[num_warp];

    sum = _warp_shuffle_reduce<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = lane_id < num_warp ? smem[lane_id] : 0;
        sum = _warp_shuffle_reduce<num_warp>(sum);
        if (tid == 0) {
            atomicAdd(b, sum);
        }
    }
}

// 将 int4 的四个 32 位分量各解释为四个 INT8，与全 1 字节向量做点积后累加。
__device__ __forceinline__ int _sum_i8x16(int4 value, int tsum) {
    const int beta = 0x01010101;
    tsum = __dp4a(value.x, beta, tsum);
    tsum = __dp4a(value.y, beta, tsum);
    tsum = __dp4a(value.z, beta, tsum);
    tsum = __dp4a(value.w, beta, tsum);
    return tsum;
}

// 每轮加载四组 16 字节到四个局部累加器，循环结束后合并为线程总和。
// N 需为 64 的倍数且地址按 16 字节对齐；这里只判断第一组的起点。
template <const int block_size = 256>
__global__ void reduce_sum_i8x64_packed_dp4a_kernel(int8_t *a, int32_t *b, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int32_t tsum[4] = {0};
    int4 value[4];
    for (int i = idx * 64; i < N; i += gridDim.x * blockDim.x * 64) {
        LDST128BITS(value[0]) = LDST128BITS(a[i]);
        LDST128BITS(value[1]) = LDST128BITS(a[i + 16]);
        LDST128BITS(value[2]) = LDST128BITS(a[i + 32]);
        LDST128BITS(value[3]) = LDST128BITS(a[i + 48]);
#pragma unroll
        for (int j = 0; j < 4; j++)
            tsum[j] = _sum_i8x16(value[j], tsum[j]);
    }
    int sum = tsum[0] + tsum[1] + tsum[2] + tsum[3];

    const int num_warp = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    __shared__ int32_t smem[num_warp];

    sum = _warp_shuffle_reduce<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = lane_id < num_warp ? smem[lane_id] : 0;
        sum = _warp_shuffle_reduce<num_warp>(sum);
        if (tid == 0) {
            atomicAdd(b, sum);
        }
    }
}

struct Generic {};

// 只检查 CUDA 和连续性；INT8 包装尤其没有验证实际 dtype，调用方必须传入 INT8。
#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

// 浮点包装最多启动 512 块；所有块原子加到清零的 FP32 输出，再转换回输入 dtype。
// 因此 FP16 输出还会产生最终舍入，不能据 FP32 累加推断返回值没有溢出。
#define binding_func_gen(name, num, element_dtype)                                                                     \
    torch::Tensor name(torch::Tensor a) {                                                                              \
        CHECK_T(a);                                                                                                    \
        const int total_elements = a.numel();                                                                          \
        const int threads_per_block = 256;                                                                             \
        const int blocks_per_grid = std::min((total_elements / num + threads_per_block - 1) / threads_per_block, 512); \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
        auto out = torch::zeros({1}, a.options().dtype(torch::kFloat32));                                              \
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(                                                                           \
            a.scalar_type(), #name, ([&] {                                                                             \
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
                    name##_kernel<threads_per_block>                                                                   \
                        <<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<CastT *>(a.data_ptr()),   \
                                                                            reinterpret_cast<float *>(out.data_ptr()), \
                                                                            total_elements);                           \
                } else {                                                                                               \
                    TORCH_CHECK(false, #name " does not support " + string(toString(a.scalar_type())));                \
                }                                                                                                      \
            }));                                                                                                       \
        return out.to(a.scalar_type());                                                                                \
    }

// 整数包装最多启动 1024 块，输出是 INT32；与默认返回 INT64 的 torch.sum 并非相同范围。
// 这里未校验 a 的 dtype，element_dtype 参数也没有用于运行时检查。
#define binding_func_gen_int(name, num, element_dtype)                                                                 \
    torch::Tensor name(torch::Tensor a) {                                                                              \
        CHECK_T(a);                                                                                                    \
        const int total_elements = a.numel();                                                                          \
        const int threads_per_block = 256;                                                                             \
        const int blocks_per_grid =                                                                                    \
            std::min((total_elements / num + threads_per_block - 1) / threads_per_block, 1024);                        \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
        auto out = torch::zeros({1}, a.options().dtype(torch::kInt32));                                                \
        name##_kernel<threads_per_block><<<blocks_per_grid, threads_per_block, 0, stream>>>(                           \
            reinterpret_cast<int8_t *>(a.data_ptr()), reinterpret_cast<int *>(out.data_ptr()), total_elements);        \
        return out;                                                                                                    \
    }

binding_func_gen(reduce_sum, 1, Generic);
binding_func_gen(reduce_sum_fp32x4, 4, float);
binding_func_gen(reduce_sum_fp16x2, 2, half);
binding_func_gen(reduce_sum_fp16x8_packed, 8, half);
binding_func_gen_int(reduce_sum_i8, 1, int);
binding_func_gen_int(reduce_sum_i8x16_packed, 16, int);
binding_func_gen_int(reduce_sum_i8x16_packed_dp4a, 16, int);
binding_func_gen_int(reduce_sum_i8x64_packed_dp4a, 64, int);

// 将 C++ 包装函数注册为 Python 扩展入口。

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(reduce_sum);
    torch_pybinding_func(reduce_sum_fp32x4);
    torch_pybinding_func(reduce_sum_fp16x2);
    torch_pybinding_func(reduce_sum_fp16x8_packed);
    torch_pybinding_func(reduce_sum_i8);
    torch_pybinding_func(reduce_sum_i8x16_packed);
    torch_pybinding_func(reduce_sum_i8x16_packed_dp4a);
    torch_pybinding_func(reduce_sum_i8x64_packed_dp4a);
}
