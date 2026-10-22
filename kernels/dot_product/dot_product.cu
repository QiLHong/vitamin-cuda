/*
 * 中文导读：对两个等长数组计算点积，线程局部累加后做块内规约，再原子加到输出。
 * FP32 工作区从零开始，返回时转换为 a 的 dtype；浮点原子加的顺序可能影响末位舍入。
 * 两输入需同 dtype、同设备且长度一致，当前包装未验证所有关系。
 * 向量版本只检查组起点，长度需是对应组大小的正整数倍；打包访问还需正确对齐。
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

// XOR 交换逐轮合并规约值；使用完整线程束掩码，调用点需保证相应线程参与。
template <const int warp_size = WARP_SIZE, typename T>
__device__ __forceinline__ T _warp_shuffle_reduce(T x) {
#pragma unroll
    for (int mask = warp_size >> 1; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask);
    }
    return x;
}

// 标量入口：模板也用于 FP16；乘法在转为 float 之前发生。
template <const int block_size = 256, typename T>
__global__ void dot_product_kernel(T *a, T *b, float *c, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        sum += static_cast<float>(a[i] * b[i]);
    }

    const int num_warp = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    __shared__ float smem[num_warp];

    // 先在线程束内规约；lane 0 发布部分和，屏障保证第一线程束能读取所有部分和。
    sum = _warp_shuffle_reduce<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    sum = lane_id < num_warp ? smem[lane_id] : 0.f;

    if (warp_id == 0) {
        sum = _warp_shuffle_reduce(sum);
        if (tid == 0) {
            atomicAdd(c, sum);
        }
    }
}

// FP32x4：每线程每轮读取四对元素，做四次乘法后加入 FP32 局部和。
// float4 要求 16 字节对齐，i<N 不会保护最后一组内的其余三个元素。
template <const int block_size = 256>
__global__ void dot_product_fp32x4_kernel(float *a, float *b, float *c, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    float4 a_val, b_val;
    for (int i = idx * 4; i < N; i += gridDim.x * blockDim.x * 4) {
        a_val = FLOAT4(a[i]);
        b_val = FLOAT4(b[i]);
        sum += a_val.x * b_val.x + a_val.y * b_val.y + a_val.z * b_val.z + a_val.w * b_val.w;
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

    sum = lane_id < num_warp ? smem[lane_id] : 0.f;

    if (warp_id == 0) {
        sum = _warp_shuffle_reduce(sum);
        if (tid == 0) {
            atomicAdd(c, sum);
        }
    }
}

// FP16x2：先在 half2 中乘法，再将乘积转为 float2 后累加；乘法已经经历半精度舍入。
// 长度需为偶数且 half2 地址按 4 字节对齐。
template <const int block_size = 256>
__global__ void dot_product_fp16x2_kernel(half *a, half *b, float *c, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    for (int i = idx * 2; i < N; i += gridDim.x * blockDim.x * 2) {
        half2 a_val = HALF2(a[i]);
        half2 b_val = HALF2(b[i]);
        float2 t = __half22float2(__hmul2(a_val, b_val));
        sum += t.x + t.y;
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

    sum = lane_id < num_warp ? smem[lane_id] : 0.f;

    if (warp_id == 0) {
        sum = _warp_shuffle_reduce(sum);
        if (tid == 0) {
            atomicAdd(c, sum);
        }
    }
}

// FP16x8：每轮搬运八个 half，分成四组 half2 计算，再用 FP32 累加。
// 长度需为 8 的倍数，128 位读写地址按 16 字节对齐；没有逐元素尾部路径。
template <const int block_size = 256>
__global__ void dot_product_fp16x8_packed_kernel(half *a, half *b, float *c, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    pack128 pack_p, pack_v;
    for (int i = idx * 8; i < N; i += gridDim.x * blockDim.x * 8) {
        pack_p.f4 = LDST128BITS(a[i]);
        pack_v.f4 = LDST128BITS(b[i]);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 t = __half22float2(__hmul2(pack_p.h2[j], pack_v.h2[j]));
            sum += t.x + t.y;
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

    sum = lane_id < num_warp ? smem[lane_id] : 0.f;

    if (warp_id == 0) {
        sum = _warp_shuffle_reduce(sum);
        if (tid == 0) {
            atomicAdd(c, sum);
        }
    }
}

struct Generic {};

// 这里只检查 CUDA 与连续性，不验证 b 的长度、dtype、设备是否与 a 一致。
#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

// 先申请并清零一个 FP32 输出，供所有块 atomicAdd；返回时再转回输入 dtype。
// 网格大小来自元素数/每线程组大小，使用 int 保存尺寸；空输入与超大输入需另行处理。
#define binding_func_gen(name, num, element_dtype)                                                                     \
    torch::Tensor name(torch::Tensor a, torch::Tensor b) {                                                             \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int total_elements = a.numel();                                                                          \
        const int threads_per_block = 256;                                                                             \
        const int blocks_per_grid =                                                                                    \
            std::min((total_elements / num + threads_per_block - 1) / threads_per_block, 1024);                        \
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
                    name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(                                  \
                        reinterpret_cast<CastT *>(a.data_ptr()),                                                       \
                        reinterpret_cast<CastT *>(b.data_ptr()),                                                       \
                        reinterpret_cast<float *>(out.data_ptr()),                                                     \
                        total_elements);                                                                               \
                } else {                                                                                               \
                    TORCH_CHECK(false, #name " does not support " + string(toString(a.scalar_type())));                \
                }                                                                                                      \
            }));                                                                                                       \
        return out.to(a.scalar_type());                                                                                \
    }

binding_func_gen(dot_product, 1, Generic);
binding_func_gen(dot_product_fp32x4, 4, float);
binding_func_gen(dot_product_fp16x2, 2, half);
binding_func_gen(dot_product_fp16x8_packed, 8, half);

// 将 C++ 包装函数注册为同名 Python 扩展入口。

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(dot_product);
    torch_pybinding_func(dot_product_fp32x4);
    torch_pybinding_func(dot_product_fp16x2);
    torch_pybinding_func(dot_product_fp16x8_packed);
}
