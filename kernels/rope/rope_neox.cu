/*
 * 中文导读：NeoX 风格的旋转位置编码，输入/输出布局为连续 FP32 [bs,seq_len,head_dim]。
 * 将第 i 维与第 i+head_dim/2 维配成 (x,y)，按位置相关角度旋转；不是相邻偶/奇维配对。
 * theta 固定为 10000，位置从 0 开始；没有额外 position_ids 或位置偏移参数。
 * 包装未完整检查形状、设备和输出大小；本次只补注释，没有修改算法或执行 GPU 测试。
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

const float theta = 10000.f;

// NeoX 风格的 RoPE：将 head_dim 的前、后两半中对应的元素配成一对进行旋转。
__global__ void rope_kernel(float *a, float *b, int seq_len, int head_dim) {
    int pos = blockIdx.x;
    int idx = blockIdx.y * (seq_len * head_dim) + pos * head_dim + threadIdx.x;
    float x = a[idx];
    float y = a[idx + (head_dim >> 1)];

    int f_idx = threadIdx.x % (head_dim >> 1);

    float inv_freq = 1.f / __powf(theta, (2.0f * f_idx) / head_dim);
    float angle = pos * inv_freq;

    float c, s;
    __sincosf(angle, &s, &c);

    float x_new = x * c - y * s;
    float y_new = y * c + x * s;
    b[idx] = x_new;
    b[idx + head_dim / 2] = y_new;
}

// 一个线程处理四对旋转元素，一个线程块覆盖四个序列位置，读写使用 float4。
// 当前网格按 seq_len/4 向下取整，且没有核内边界检查，因此要求 seq_len 为 4 的倍数。
// head_dim/2 还须为 4 的倍数，即 head_dim 为 8 的倍数，保证组内不跨行与 16 字节对齐。
__global__ void rope_fp32x4_kernel(float *a, float *b, int seq_len, int head_dim) {
    int pos = blockIdx.x * 4;
    int tid = threadIdx.x;
    pos += tid * 4 / (head_dim >> 1);
    int f_idx = tid * 4 % (head_dim >> 1);
    int idx = blockIdx.y * (seq_len * head_dim) + pos * head_dim + f_idx;
    float4 x = LDST128BITS(a[idx]);
    float4 y = LDST128BITS(a[idx + (head_dim >> 1)]);
    float inv_freq[4], c[4], s[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        inv_freq[i] = 1.f / __powf(theta, (2.0f * (f_idx + i)) / head_dim);
        __sincosf(pos * inv_freq[i], &s[i], &c[i]);
    }

    float4 x_new, y_new;
    x_new.x = x.x * c[0] - y.x * s[0];
    x_new.y = x.y * c[1] - y.y * s[1];
    x_new.z = x.z * c[2] - y.z * s[2];
    x_new.w = x.w * c[3] - y.w * s[3];
    y_new.x = y.x * c[0] + x.x * s[0];
    y_new.y = y.y * c[1] + x.y * s[1];
    y_new.z = y.z * c[2] + x.z * s[2];
    y_new.w = y.w * c[3] + x.w * s[3];

    LDST128BITS(b[idx]) = x_new;
    LDST128BITS(b[idx + head_dim / 2]) = y_new;
}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

// 每个块启动 head_dim/2 个线程，要求正偶数 head_dim 且线程数不超过设备每块上限。
// a/b 应同形、同设备、连续 FP32；原包装只显式检查 CUDA/连续性及 data_ptr<float>() 类型。
// num 控制一个块覆盖的序列位置数：标量版本为 1，向量化版本为 4。
#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int bs = a.size(0);                                                                                      \
        const int seq_len = a.size(1);                                                                                 \
        const int head_dim = a.size(2);                                                                                \
        const int threads_per_block = head_dim / 2;                                                                    \
        const dim3 blocks_per_grid(seq_len / num, bs);                                                                 \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(                                              \
            a.data_ptr<float>(), b.data_ptr<float>(), seq_len, head_dim);                                              \
    }

binding_func_gen(rope, 1, float);
binding_func_gen(rope_fp32x4, 4, float);
// Python 绑定。

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(rope);
    torch_pybinding_func(rope_fp32x4);
}
