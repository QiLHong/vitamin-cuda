/*
 * 中文导读：行主序 FP32 GEMM，C[M,N] = A[M,K] @ B[K,N]。
 * 朴素版本每线程一个输出；分块版本每块 128×128，每线程累加 8×8 个输出。
 * A 的共享内存转置改善内层读取，swizzle 调整 Bank 映射，rw 调整结果写回，dbf 引入双缓冲。
 * 分块版本没有通用尾部保护：入门采用 M/N 为 128、K 为 16 的正整数倍，且宽访问按 16 字节对齐。
 * 所有张量须同设备、同形状约定、连续、FP32 且无有害重叠；包装未逐项验证。
 * 下面只增加阅读说明，原有线程映射、计算、同步和绑定保持不变。
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

#define SWIZZLE_A(x, y) ((y) ^ ((x >> 2) << 3))

// FP32 矩阵乘法：每个线程计算一个输出元素。
__global__ void sgemm_naive_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.f;
    if (row < m && col < n) {
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

// 每个线程块计算 C 的 128×128 数据块，每个线程计算其中的 8×8 数据块。
template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_tiling_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255；共 8 个线程束，按 2×4 排布；每个线程束内的线程按 8×4 排布。
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 每块每次加载 A 的 64×16 和 B 的 8×128；两次分别覆盖完整的 128×16 和 16×128。
    // 每 4 个线程加载 A 的一行（16 个元素）；每 32 个线程加载 B 的一行（128 个元素）。
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    int load_b_row = tid / WARP_SIZE;       // 0~7（原注释写 0~8；256 个线程除以 32 的实际范围为 0~7）。
    int load_b_col = (tid % WARP_SIZE) * 4; // 0,4,8,12,16,20,24,28...

    // 线程束分块：每 4 个线程束覆盖 C 的上半或下半个 64×128 数据块。
    int warp_row = warp_id / 4;      // 0, 1
    int warp_col = warp_id % 4;      // 0, 1, 2, 3
    int t_row_in_warp = lane_id / 4; // 0~7
    int t_col_in_warp = lane_id % 4; // 0~3

    // C 的输出起点：每线程计算 8×8 个元素，256×64=128×128。
    int c_row = warp_row * 64 + t_row_in_warp * 8;
    int c_col = warp_col * 32 + t_col_in_warp * 8;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float sum[TM][TN] = {0.f};

    for (int bk = 0; bk < k; bk += BK) {
        FLOAT4(As[load_a_row][load_a_col]) = FLOAT4(a[(by * BM + load_a_row) * k + bk + load_a_col]);
        FLOAT4(As[load_a_row + 64][load_a_col]) = FLOAT4(a[(by * BM + load_a_row + 64) * k + bk + load_a_col]);

        FLOAT4(Bs[load_b_row][load_b_col]) = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        FLOAT4(Bs[load_b_row + 8][load_b_col]) = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        __syncthreads();

        // 累加一个 8×8 外积。
#pragma unroll
        for (int i = 0; i < BK; i++) {
            float reg_a[TM], reg_b[TN];

#pragma unroll
            for (int m_idx = 0; m_idx < TM; ++m_idx)
                reg_a[m_idx] = As[c_row + m_idx][i];

            FLOAT4(reg_b[0]) = FLOAT4(Bs[i][c_col]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[i][c_col + 4]);

#pragma unroll
            for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < TN; ++n_idx) {
                    sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
                }
            }
        }
        __syncthreads();
    }

    // 写回 C。
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col]) = FLOAT4(sum[i][0]);
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col + 4]) = FLOAT4(sum[i][4]);
    }
}

// 每个线程块计算 C 的 128×128 数据块，每个线程计算其中的 8×8 数据块。
template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_at_tiling_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255；共 8 个线程束，按 2×4 排布；每个线程束内的线程按 8×4 排布。
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 每块每次加载 A 的 64×16 和 B 的 8×128，总共加载两次。
    // 每 4 个线程加载 A 的一行（16 个元素）；每 32 个线程加载 B 的一行（128 个元素）。
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    int load_b_row = tid / WARP_SIZE;       // 0~7（原注释写 0~8；256 个线程除以 32 的实际范围为 0~7）。
    int load_b_col = (tid % WARP_SIZE) * 4; // 0,4,8,12,16,20,24,28...

    // 在线程块内部，继续按线程束划分输出数据块。
    int warp_row = warp_id / 4;      // 0, 1
    int warp_col = warp_id % 4;      // 0, 1, 2, 3
    int t_row_in_warp = lane_id / 4; // 0~7
    int t_col_in_warp = lane_id % 4; // 0~3

    // C 的输出起点：每线程计算 8×8 个元素，256×64=128×128。
    int c_row = warp_row * 64 + t_row_in_warp * 8;
    int c_col = warp_col * 32 + t_col_in_warp * 8;

    __shared__ float As_T[BK][BM];
    __shared__ float Bs[BK][BN];

    float sum[TM][TN] = {0.f};

    for (int bk = 0; bk < k; bk += BK) {
        // 将 A 转置后存入共享内存。
        float4 tmp_a0 = FLOAT4(a[(by * BM + load_a_row) * k + bk + load_a_col]);
        As_T[load_a_col + 0][load_a_row] = tmp_a0.x;
        As_T[load_a_col + 1][load_a_row] = tmp_a0.y;
        As_T[load_a_col + 2][load_a_row] = tmp_a0.z;
        As_T[load_a_col + 3][load_a_row] = tmp_a0.w;

        float4 tmp_a1 = FLOAT4(a[(by * BM + load_a_row + 64) * k + bk + load_a_col]);
        As_T[load_a_col + 0][load_a_row + 64] = tmp_a1.x;
        As_T[load_a_col + 1][load_a_row + 64] = tmp_a1.y;
        As_T[load_a_col + 2][load_a_row + 64] = tmp_a1.z;
        As_T[load_a_col + 3][load_a_row + 64] = tmp_a1.w;

        FLOAT4(Bs[load_b_row][load_b_col]) = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        FLOAT4(Bs[load_b_row + 8][load_b_col]) = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        __syncthreads();

        // 累加一个 8×8 外积。
#pragma unroll
        for (int i = 0; i < BK; i++) {
            float reg_a[TM], reg_b[TN];

            FLOAT4(reg_a[0]) = FLOAT4(As_T[i][c_row]);
            FLOAT4(reg_a[4]) = FLOAT4(As_T[i][c_row + 4]);

            FLOAT4(reg_b[0]) = FLOAT4(Bs[i][c_col]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[i][c_col + 4]);

#pragma unroll
            for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < TN; ++n_idx) {
                    sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
                }
            }
        }
        __syncthreads();
    }

    // 写回 C。
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col]) = FLOAT4(sum[i][0]);
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col + 4]) = FLOAT4(sum[i][4]);
    }
}

// 通过 swizzle 重排共享内存地址，以避免此访问模式中的 Bank 冲突。
template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_at_bcf_swizzling_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255；共 8 个线程束，按 2×4 排布；每个线程束内的线程按 8×4 排布。
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 每块每次加载 A 的 64×16 和 B 的 8×128，总共加载两次。
    // 每 4 个线程加载 A 的一行（16 个元素）；每 32 个线程加载 B 的一行（128 个元素）。
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    int load_b_row = tid / WARP_SIZE;       // 0~7（原注释写 0~8；256 个线程除以 32 的实际范围为 0~7）。
    int load_b_col = (tid % WARP_SIZE) * 4; // 0,4,8,12,16,20,24,28...

    // 在线程块内部，继续按线程束划分输出数据块。
    int warp_row = warp_id / 4;      // 0, 1
    int warp_col = warp_id % 4;      // 0, 1, 2, 3
    int t_row_in_warp = lane_id / 4; // 0~7
    int t_col_in_warp = lane_id % 4; // 0~3

    // C 的输出起点：每线程计算 8×8 个元素，256×64=128×128。
    int c_row = warp_row * 64 + t_row_in_warp * 8;
    int c_col = warp_col * 32 + t_col_in_warp * 8;

    __shared__ float As_T[BK][BM];
    __shared__ float Bs[BK][BN];

    float sum[TM][TN] = {0.f};

    for (int bk = 0; bk < k; bk += BK) {
        // 将 A 转置，同时用 swizzle 重排共享内存地址。
        float4 tmp_a0 = FLOAT4(a[(by * BM + load_a_row) * k + bk + load_a_col]);
        As_T[load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = tmp_a0.x;
        As_T[load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = tmp_a0.y;
        As_T[load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = tmp_a0.z;
        As_T[load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = tmp_a0.w;

        float4 tmp_a1 = FLOAT4(a[(by * BM + load_a_row + 64) * k + bk + load_a_col]);
        As_T[load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = tmp_a1.x;
        As_T[load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = tmp_a1.y;
        As_T[load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = tmp_a1.z;
        As_T[load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = tmp_a1.w;

        FLOAT4(Bs[load_b_row][load_b_col]) = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        FLOAT4(Bs[load_b_row + 8][load_b_col]) = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        __syncthreads();

        // 累加一个 8×8 外积。
#pragma unroll
        for (int i = 0; i < BK; i++) {
            float reg_a[TM], reg_b[TN];

            FLOAT4(reg_a[0]) = FLOAT4(As_T[i][SWIZZLE_A(i, c_row)]);
            FLOAT4(reg_a[4]) = FLOAT4(As_T[i][SWIZZLE_A(i, c_row + 4)]);

            FLOAT4(reg_b[0]) = FLOAT4(Bs[i][c_col]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[i][c_col + 4]);

#pragma unroll
            for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < TN; ++n_idx) {
                    sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
                }
            }
        }
        __syncthreads();
    }

    // 写回 C。
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col]) = FLOAT4(sum[i][0]);
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col + 4]) = FLOAT4(sum[i][4]);
    }
}

// 共享内存 swizzle 避免 Bank 冲突，并合并 C 的全局内存写回。
template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_at_bcf_swizzling_rw_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255；共 8 个线程束。
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 每块每次加载 A 的 64×16 和 B 的 8×128，总共加载两次。
    // 每 4 个线程加载 A 的一行（16 个元素）；每 32 个线程加载 B 的一行（128 个元素）。
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    int load_b_row = tid / WARP_SIZE;       // 0~7（原注释写 0~8；256 个线程除以 32 的实际范围为 0~7）。
    int load_b_col = (tid % WARP_SIZE) * 4; // 0,4,8,12,16,20,24,28...

    // 线程在所属线程束内的行偏移仍然为 0 或 8。
    int t_row_in_warp = (lane_id / 16) * 8;

    // 每个线程束覆盖 16 行；每 16 个线程处理 8 行×128 列，每线程分两次访问 float4。
    // 例如 T0 读写第 0~3 列和第 64~67 列；连续 8 个线程在每次写回中覆盖 32 个连续
    // float（128 字节），从地址分布上形成合并访存；实际事务数仍取决于对齐和硬件。
    int c_row = warp_id * 16 + t_row_in_warp;
    int c_col_base = (lane_id % 16) * 4;
    int c_col_0 = c_col_base;      // 0~3
    int c_col_1 = c_col_base + 64; // 64~67

    __shared__ float As_T[BK][BM];
    __shared__ float Bs[BK][BN];

    float sum[TM][TN] = {0.f};

    for (int bk = 0; bk < k; bk += BK) {
        // 将 A 转置，同时用 swizzle 重排共享内存地址。
        float4 tmp_a0 = FLOAT4(a[(by * BM + load_a_row) * k + bk + load_a_col]);
        As_T[load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = tmp_a0.x;
        As_T[load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = tmp_a0.y;
        As_T[load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = tmp_a0.z;
        As_T[load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = tmp_a0.w;

        float4 tmp_a1 = FLOAT4(a[(by * BM + load_a_row + 64) * k + bk + load_a_col]);
        As_T[load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = tmp_a1.x;
        As_T[load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = tmp_a1.y;
        As_T[load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = tmp_a1.z;
        As_T[load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = tmp_a1.w;

        FLOAT4(Bs[load_b_row][load_b_col]) = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        FLOAT4(Bs[load_b_row + 8][load_b_col]) = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        __syncthreads();

        // 累加一个 8×8 外积。
#pragma unroll
        for (int i = 0; i < BK; i++) {
            float reg_a[TM], reg_b[TN];

            FLOAT4(reg_a[0]) = FLOAT4(As_T[i][SWIZZLE_A(i, c_row)]);
            FLOAT4(reg_a[4]) = FLOAT4(As_T[i][SWIZZLE_A(i, c_row + 4)]);

            FLOAT4(reg_b[0]) = FLOAT4(Bs[i][c_col_0]); // 当前线程的第一组四列；T0 对应第 0~3 列。
            FLOAT4(reg_b[4]) = FLOAT4(Bs[i][c_col_1]); // 当前线程的第二组四列；T0 对应第 64~67 列。

#pragma unroll
            for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < TN; ++n_idx) {
                    sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col_0]) = FLOAT4(sum[i][0]);
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col_1]) = FLOAT4(sum[i][4]);
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_at_bcf_swizzling_dbf_rw_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255；共 8 个线程束。
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 数据加载的线程映射。
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    int load_b_row = tid / WARP_SIZE;       // 0~7（原注释写 0~8；256 个线程除以 32 的实际范围为 0~7）。
    int load_b_col = (tid % WARP_SIZE) * 4; // 0,4,8,12,16,20,24,28...

    // C 的计算、读取和写回映射，与上一版本相同。
    int t_row_in_warp = (lane_id / 16) * 8;
    int c_row = warp_id * 16 + t_row_in_warp;
    int c_col_base = (lane_id % 16) * 4;
    int c_col_0 = c_col_base; // 0~3
    // int c_col_1 = c_col_base + 64; // 64~67

    // 双缓冲。
    __shared__ float As_T[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    float sum[TM][TN] = {0.f};

    // 用全局内存的一维指针，方便沿流水线推进数据位置。
    float *a_ptr = a + (by * BM + load_a_row) * k + load_a_col;
    // float *a_ptr_64 = a + (by * BM + load_a_row + 64) * k + load_a_col;
    float *b_ptr = b + load_b_row * n + bx * BN + load_b_col;
    // float *b_ptr_8 = b + (load_b_row + 8) * n + bx * BN + load_b_col;

    // 预取第一个数据块需额外暂存 16 个 float；上面省略两条指针是原作者控制寄存器压力的做法，实际寄存器数须看编译结果。
    float4 tmp_a0 = FLOAT4(a_ptr[0]);
    float4 tmp_a1 = FLOAT4(a_ptr[64 * k]);
    float4 tmp_b0 = FLOAT4(b_ptr[0]);
    float4 tmp_b1 = FLOAT4(b_ptr[8 * n]);

    As_T[0][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = tmp_a0.x;
    As_T[0][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = tmp_a0.y;
    As_T[0][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = tmp_a0.z;
    As_T[0][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = tmp_a0.w;

    As_T[0][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = tmp_a1.x;
    As_T[0][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = tmp_a1.y;
    As_T[0][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = tmp_a1.z;
    As_T[0][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = tmp_a1.w;

    FLOAT4(Bs[0][load_b_row][load_b_col]) = tmp_b0;
    FLOAT4(Bs[0][load_b_row + 8][load_b_col]) = tmp_b1;

    __syncthreads();

    // 双缓冲的读、写索引。
    int write_idx = 1;
    int read_idx = 0;
    // 主循环。
    for (int bk = BK; bk < k; bk += BK) {
        // 沿 K 维推进指针。
        a_ptr += BK;
        b_ptr += BK * n;

        // 预取下一个数据块：发出全局内存加载后，接着计算当前数据块。
        tmp_a0 = FLOAT4(a_ptr[0]);
        tmp_a1 = FLOAT4(a_ptr[64 * k]);
        tmp_b0 = FLOAT4(b_ptr[0]);
        tmp_b1 = FLOAT4(b_ptr[8 * n]);

        // 计算过程与没有流水线的版本相同。
#pragma unroll
        for (int i = 0; i < BK; i++) {
            float reg_a[TM], reg_b[TN];

            FLOAT4(reg_a[0]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row)]);
            FLOAT4(reg_a[4]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row + 4)]);

            FLOAT4(reg_b[0]) = FLOAT4(Bs[read_idx][i][c_col_0]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[read_idx][i][c_col_0 + 64]);

#pragma unroll
            for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < TN; ++n_idx) {
                    sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
                }
            }
        }

        // 当前计算结束，将预取到寄存器的数据放入共享内存的写缓冲。
        As_T[write_idx][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = tmp_a0.x;
        As_T[write_idx][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = tmp_a0.y;
        As_T[write_idx][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = tmp_a0.z;
        As_T[write_idx][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = tmp_a0.w;

        As_T[write_idx][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = tmp_a1.x;
        As_T[write_idx][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = tmp_a1.y;
        As_T[write_idx][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = tmp_a1.z;
        As_T[write_idx][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = tmp_a1.w;

        FLOAT4(Bs[write_idx][load_b_row][load_b_col]) = tmp_b0;
        FLOAT4(Bs[write_idx][load_b_row + 8][load_b_col]) = tmp_b1;

        __syncthreads();
        write_idx ^= 1;
        read_idx ^= 1;
    }
    // 处理最后一个已经预取的数据块。
#pragma unroll
    for (int i = 0; i < BK; i++) {
        float reg_a[TM], reg_b[TN];

        FLOAT4(reg_a[0]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row)]);
        FLOAT4(reg_a[4]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row + 4)]);

        FLOAT4(reg_b[0]) = FLOAT4(Bs[read_idx][i][c_col_0]);
        FLOAT4(reg_b[4]) = FLOAT4(Bs[read_idx][i][c_col_0 + 64]);

#pragma unroll
        for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
            for (int n_idx = 0; n_idx < TN; ++n_idx) {
                sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
            }
        }
    }
    // 流水线结束，写回 C。
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col_0]) = FLOAT4(sum[i][0]);
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col_0 + 64]) = FLOAT4(sum[i][4]);
    }
}

// 这里只检查 CUDA 与连续性；没有完整验证设备一致、形状、dtype、尾部及对齐。
// 下面的宏负责在主机端取指针、计算网格并使用 PyTorch 当前流；启动通常是异步的。
#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a, torch::Tensor b, torch::Tensor c) {                                                     \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        CHECK_T(c);                                                                                                    \
        const int M = a.size(0);                                                                                       \
        const int K = a.size(1);                                                                                       \
        const int N = b.size(1);                                                                                       \
        const dim3 threads_per_block(16, 16);                                                                          \
        const dim3 blocks_per_grid((N + 15) / 16, (M + 15) / 16);                                                      \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(                                              \
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);                                   \
    }

#define binding_tiled_func_gen(name)                                                                                   \
    void name(torch::Tensor a, torch::Tensor b, torch::Tensor c) {                                                     \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        CHECK_T(c);                                                                                                    \
        const int M = a.size(0);                                                                                       \
        const int K = a.size(1);                                                                                       \
        const int N = b.size(1);                                                                                       \
        const int BM = 128;                                                                                            \
        const int BN = 128;                                                                                            \
        const int threads_per_block = 256;                                                                             \
        const dim3 blocks_per_grid((N + BN - 1) / BN, (M + BM - 1) / BM);                                              \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<128, 128, 16, 8, 8><<<blocks_per_grid, threads_per_block, 0, stream>>>(                          \
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);                                   \
    }

extern void sgemm_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c);
extern void sgemm_cublas_tf32(torch::Tensor a, torch::Tensor b, torch::Tensor c);

extern void sgemm_tf32_bt(torch::Tensor a, torch::Tensor b, torch::Tensor c);
extern void sgemm_tf32_bt_swizzle(torch::Tensor a, torch::Tensor b, torch::Tensor c);
extern void sgemm_tf32_bt_swizzle_dbf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
extern void sgemm_tf32_swizzle_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
extern void sgemm_tf32_swizzle_bcf_dbf(torch::Tensor a, torch::Tensor b, torch::Tensor c);

binding_func_gen(sgemm_naive, 1, float);

binding_tiled_func_gen(sgemm_tiling);
binding_tiled_func_gen(sgemm_at_tiling);
binding_tiled_func_gen(sgemm_at_bcf_swizzling);
binding_tiled_func_gen(sgemm_at_bcf_swizzling_rw);
binding_tiled_func_gen(sgemm_at_bcf_swizzling_dbf_rw);

// 导出供 Python 调用的绑定。

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(sgemm_cublas);
    torch_pybinding_func(sgemm_naive);
    torch_pybinding_func(sgemm_tiling);
    torch_pybinding_func(sgemm_at_tiling);
    torch_pybinding_func(sgemm_at_bcf_swizzling);
    torch_pybinding_func(sgemm_at_bcf_swizzling_rw);
    torch_pybinding_func(sgemm_at_bcf_swizzling_dbf_rw);
    // Tensor Core 路径。
    torch_pybinding_func(sgemm_cublas_tf32);
    torch_pybinding_func(sgemm_tf32_bt);
    torch_pybinding_func(sgemm_tf32_bt_swizzle);
    torch_pybinding_func(sgemm_tf32_bt_swizzle_dbf);
    torch_pybinding_func(sgemm_tf32_swizzle_bcf);
    torch_pybinding_func(sgemm_tf32_swizzle_bcf_dbf);
}
