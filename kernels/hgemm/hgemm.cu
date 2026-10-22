/*
 * 中文导读：C[M,N] = A[M,K] @ B[K,N]，张量按连续行主序解释。
 * FP16/BF16 输入经 MMA 在 FP32 中累加，写出时再转换到输入类型。
 * 四个版本逐步加入共享内存 swizzle、双缓冲与合并写回；naive 也使用 Tensor Core。
 * 当前分块为 128×128×32，所有版本都有宽度为 8 的网格重排，且没有完整尾部保护。
 * 入门采用 M 为 128、N 为 1024、K 为 32 的正整数倍，宽访问还需满足 16 字节对齐。
 * 包装未完整校验同设备、同 dtype、形状和重叠关系；非 FP16 分支直接解释成 BF16。
 * 这些注释说明源码约定，不表示已编译、通过正确性测试或复现原文性能。
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

#define SWIZZLE_A(row, col) ((col) ^ (((row >> 1) & 0x3) << 3))

#define SWIZZLE_B(row, col) ((col) ^ (((row) & 0x7) << 3))

#define SWIZZLE_C(row, col) ((col) ^ (((row) & 0x7) << 3))

// ---------------- 内联 PTX 汇编宏 ----------------
// cp.async：将全局内存 src 中的 16 字节异步拷贝到共享内存 dst_smem_32b。
#define CP_ASYNC_CG(dst_smem_32b, src_global_ptr)                                                                      \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst_smem_32b), "l"(src_global_ptr))

// commit_group 将本线程此前的异步拷贝组成一组；wait_group 0 等待这些组完成。
// wait_group 不代替线程块屏障，后续跨线程使用共享内存时仍须协调。
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP_0() asm volatile("cp.async.wait_group 0;\n" ::)

// ldmatrix：由线程束协作，将共享内存矩阵片段加载到各线程的寄存器。
#define LDMATRIX_X4(R0, R1, R2, R3, PTR)                                                                               \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"                                    \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                                              \
                 : "r"(PTR))

#define LDMATRIX_X2(R0, R1, PTR)                                                                                       \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(R0), "=r"(R1) : "r"(PTR))

// 加载两个 8×8 数据块，并按转置布局分发到线程束寄存器。
#define LDMATRIX_X2_TRANS(R0, R1, PTR)                                                                                 \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(R0), "=r"(R1) : "r"(PTR))

// mma.sync：线程束协作执行矩阵乘加，下面以 FP32 累加 FP16/BF16 乘积。
#define M16N8K16_F16(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1)                                                           \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                                                  \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                                         \
                 : "=f"(C0), "=f"(C1), "=f"(C2), "=f"(C3)                                                              \
                 : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), "f"(C0), "f"(C1), "f"(C2), "f"(C3))

#define M16N8K16_BF16(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1)                                                          \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "                                                \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                                         \
                 : "=f"(C0), "=f"(C1), "=f"(C2), "=f"(C3)                                                              \
                 : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), "f"(C0), "f"(C1), "f"(C2), "f"(C3))

// ------------------------------------------ ldmatrix 加载与 MMA 计算 ------------------------------------------

// 一个线程块计算 C 的 128×128 数据块。
template <const int BM = 128, const int BN = 128, const int BK = 32, typename T>
__global__ void hgemm_naive_kernel(T *a, T *b, T *c, int m, int n, int k) {
    // 重排线程块到输出数据块的映射（网格 swizzle）。
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // 网格重排的逻辑分组宽度为 8。

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 全局内存到共享内存的加载索引。
    int load_a_row = tid / 4;        // 0~63
    int load_a_col = (tid % 4) * 8;  // 0,8,16,24
    int load_b_row = tid / 16;       // 0..15（K 维）
    int load_b_col = (tid % 16) * 8; // 0,8,...,120（N 维）

    // A、B 都保持便于连续加载的行主序布局。
    __shared__ T As[BM][BK];
    __shared__ T Bs[BK][BN];

    // 在线程块内部，继续按线程束划分输出数据块。
    // 每个线程束负责 C 的 64×32 数据块。
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 每线程的累加器：M 方向 4 个片段 × N 方向 4 个片段 × 每片段 4 个 float，共 64 个。
    float sum[4][4][4] = {0.f};

    // 沿 K 维遍历数据块的主循环。
    for (int bk = 0; bk < k; bk += BK) {

        // 1. 用 cp.async 异步加载 A。
        uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][load_a_col]));
        uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row + 64][load_a_col]));

        T *global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        T *global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        // 2. 用 cp.async 异步加载 B。
        uint32_t smem_b0 = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_row][load_b_col]));
        uint32_t smem_b1 = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_row + 16][load_b_col]));

        T *global_b0 = &b[(bk + load_b_row) * n + bx * BN + load_b_col];
        T *global_b1 = &b[(bk + load_b_row + 16) * n + bx * BN + load_b_col];

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // 3. Tensor Core 计算：K 维分两步，每步处理 16 个元素。
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 为 A 发出四次 ldmatrix 加载，覆盖 4×16=64 行。
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                // ldmatrix x4 的四个 8×8 子块在此共同组成一个 16×16 数据块。
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 8;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][a_col]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // 为 B 发出四次 ldmatrix 加载，覆盖 4×8=32 列。
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // 线程束内编号 0~15 的线程提供 16 行的起始地址，对应两个 8×8 子块。
                int b_row = k_offset + (lane_id % 16);
                int b_col = warp_id_n * 32 + n_idx * 8;

                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][b_col]));
                LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA 主体：在 M、N 方向各循环 4 次，执行 m16n8k16 矩阵乘加。
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    if constexpr (std::is_same_v<T, __half>) {
                        M16N8K16_F16(sum[m_idx][n_idx][0],
                                     sum[m_idx][n_idx][1],
                                     sum[m_idx][n_idx][2],
                                     sum[m_idx][n_idx][3],
                                     reg_a[m_idx][0],
                                     reg_a[m_idx][1],
                                     reg_a[m_idx][2],
                                     reg_a[m_idx][3],
                                     reg_b[n_idx][0],
                                     reg_b[n_idx][1]);
                    } else {
                        M16N8K16_BF16(sum[m_idx][n_idx][0],
                                      sum[m_idx][n_idx][1],
                                      sum[m_idx][n_idx][2],
                                      sum[m_idx][n_idx][3],
                                      reg_a[m_idx][0],
                                      reg_a[m_idx][1],
                                      reg_a[m_idx][2],
                                      reg_a[m_idx][3],
                                      reg_b[n_idx][0],
                                      reg_b[n_idx][1]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---------------- 写回 C ----------------
    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;
            int idx_0 = (c_base_row + t_row) * n + c_base_col + t_col;
            int idx_2 = (c_base_row + t_row + 8) * n + c_base_col + t_col;
            if constexpr (std::is_same_v<T, __half>) {
                HALF2(c[idx_0]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][0]));
                HALF2(c[idx_2]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][2]));
            } else {
                BFLOAT2(c[idx_0]) = __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][0]));
                BFLOAT2(c[idx_2]) = __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][2]));
            }
        }
    }
}

// 一个线程块计算 C 的 128×128 数据块。
template <const int BM = 128, const int BN = 128, const int BK = 32, typename T>
__global__ void hgemm_bcf_kernel(T *a, T *b, T *c, int m, int n, int k) {
    // 重排线程块到输出数据块的映射（网格 swizzle）。
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // 网格重排的逻辑分组宽度为 8。

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 全局内存到共享内存的加载索引。
    int load_a_row = tid / 4;        // 0~63
    int load_a_col = (tid % 4) * 8;  // 0,8,16,24
    int load_b_row = tid / 16;       // 0..15（K 维）
    int load_b_col = (tid % 16) * 8; // 0,8,...,120（N 维）

    // A、B 都保持便于连续加载的行主序布局。
    __shared__ T As[BM][BK];
    __shared__ T Bs[BK][BN];

    // 在线程块内部，继续按线程束划分输出数据块。
    // 每个线程束负责 C 的 64×32 数据块。
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 每线程的累加器：M 方向 4 个片段 × N 方向 4 个片段 × 每片段 4 个 float，共 64 个。
    float sum[4][4][4] = {0.f};

    // 沿 K 维遍历数据块的主循环。
    for (int bk = 0; bk < k; bk += BK) {

        // 1. 用 cp.async 异步加载 A。
        uint32_t smem_a0 =
            static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
        uint32_t smem_a1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

        T *global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        T *global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        // 2. 用 cp.async 异步加载 B。
        uint32_t smem_b0 =
            static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
        uint32_t smem_b1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));

        T *global_b0 = &b[(bk + load_b_row) * n + bx * BN + load_b_col];
        T *global_b1 = &b[(bk + load_b_row + 16) * n + bx * BN + load_b_col];

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // 3. Tensor Core 计算：K 维分两步，每步处理 16 个元素。
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 为 A 发出四次 ldmatrix 加载，覆盖 4×16=64 行。
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                // ldmatrix x4 的四个 8×8 子块在此共同组成一个 16×16 数据块。
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 8;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // 为 B 发出四次 ldmatrix 加载，覆盖 4×8=32 列。
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // 线程束内编号 0~15 的线程提供 16 行的起始地址，对应两个 8×8 子块。
                int b_row = k_offset + (lane_id % 16);
                int b_col = warp_id_n * 32 + n_idx * 8;

                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][SWIZZLE_B(b_row, b_col)]));
                LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA 主体：在 M、N 方向各循环 4 次，执行 m16n8k16 矩阵乘加。
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    if constexpr (std::is_same_v<T, __half>) {
                        M16N8K16_F16(sum[m_idx][n_idx][0],
                                     sum[m_idx][n_idx][1],
                                     sum[m_idx][n_idx][2],
                                     sum[m_idx][n_idx][3],
                                     reg_a[m_idx][0],
                                     reg_a[m_idx][1],
                                     reg_a[m_idx][2],
                                     reg_a[m_idx][3],
                                     reg_b[n_idx][0],
                                     reg_b[n_idx][1]);
                    } else {
                        M16N8K16_BF16(sum[m_idx][n_idx][0],
                                      sum[m_idx][n_idx][1],
                                      sum[m_idx][n_idx][2],
                                      sum[m_idx][n_idx][3],
                                      reg_a[m_idx][0],
                                      reg_a[m_idx][1],
                                      reg_a[m_idx][2],
                                      reg_a[m_idx][3],
                                      reg_b[n_idx][0],
                                      reg_b[n_idx][1]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---------------- 写回 C ----------------
    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;
            int idx_0 = (c_base_row + t_row) * n + c_base_col + t_col;
            int idx_2 = (c_base_row + t_row + 8) * n + c_base_col + t_col;
            if constexpr (std::is_same_v<T, __half>) {
                HALF2(c[idx_0]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][0]));
                HALF2(c[idx_2]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][2]));
            } else {
                BFLOAT2(c[idx_0]) = __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][0]));
                BFLOAT2(c[idx_2]) = __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][2]));
            }
        }
    }
}

// 一个线程块计算 C 的 128×128 数据块。
template <const int BM = 128, const int BN = 128, const int BK = 32, typename T>
__global__ void hgemm_bcf_dbf_kernel(T *a, T *b, T *c, int m, int n, int k) {
    // 重排线程块到输出数据块的映射（网格 swizzle）。
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // 网格重排的逻辑分组宽度为 8。

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 全局内存到共享内存的加载索引。
    int load_a_row = tid / 4;        // 0~63
    int load_a_col = (tid % 4) * 8;  // 0,8,16,24
    int load_b_row = tid / 16;       // 0..15（K 维）
    int load_b_col = (tid % 16) * 8; // 0,8,...,120（N 维）

    // A、B 都保持便于连续加载的行主序布局。
    __shared__ T As[2][BM][BK];
    __shared__ T Bs[2][BK][BN];

    // 在线程块内部，继续按线程束划分输出数据块。
    // 每个线程束负责 C 的 64×32 数据块。
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 每线程的累加器：M 方向 4 个片段 × N 方向 4 个片段 × 每片段 4 个 float，共 64 个。
    float sum[4][4][4] = {0.f};

    // ----------------------------- 流水线序言：先预取一组 As/Bs。
    // 用 cp.async 异步加载 A。
    uint32_t smem_a0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&As[0][load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
    uint32_t smem_a1 = static_cast<uint32_t>(
        __cvta_generic_to_shared(&As[0][load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

    T *global_a0 = &a[(by * BM + load_a_row) * k + load_a_col];
    T *global_a1 = &a[(by * BM + load_a_row + 64) * k + load_a_col];

    CP_ASYNC_CG(smem_a0, global_a0);
    CP_ASYNC_CG(smem_a1, global_a1);

    // 用 cp.async 异步加载 B。
    uint32_t smem_b0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[0][load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
    uint32_t smem_b1 = static_cast<uint32_t>(
        __cvta_generic_to_shared(&Bs[0][load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));

    T *global_b0 = &b[(load_b_row)*n + bx * BN + load_b_col];
    T *global_b1 = &b[(load_b_row + 16) * n + bx * BN + load_b_col];

    CP_ASYNC_CG(smem_b0, global_b0);
    CP_ASYNC_CG(smem_b1, global_b1);

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP_0();
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    // 沿 K 维遍历数据块的主循环。
    for (int bk = 32; bk < k; bk += BK) {

        // 1. 用 cp.async 异步加载 A。
        smem_a0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[write_idx][load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
        smem_a1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[write_idx][load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

        // 按固定步长推进全局内存指针，减少反复构造地址所需的中间变量。
        global_a0 += BK;
        global_a1 += BK;

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        // 2. 用 cp.async 异步加载 B。
        smem_b0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[write_idx][load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
        smem_b1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[write_idx][load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));

        global_b0 += BK * n;
        global_b1 += BK * n;

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);

        CP_ASYNC_COMMIT_GROUP();

        // 3. Tensor Core 计算：K 维分两步，每步处理 16 个元素。
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 为 A 发出四次 ldmatrix 加载，覆盖 4×16=64 行。
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                // ldmatrix x4 的四个 8×8 子块在此共同组成一个 16×16 数据块。
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 8;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // 为 B 发出四次 ldmatrix 加载，覆盖 4×8=32 列。
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // 线程束内编号 0~15 的线程提供 16 行的起始地址，对应两个 8×8 子块。
                int b_row = k_offset + (lane_id % 16);
                int b_col = warp_id_n * 32 + n_idx * 8;

                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
                LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA 主体：在 M、N 方向各循环 4 次，执行 m16n8k16 矩阵乘加。
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    if constexpr (std::is_same_v<T, __half>) {
                        M16N8K16_F16(sum[m_idx][n_idx][0],
                                     sum[m_idx][n_idx][1],
                                     sum[m_idx][n_idx][2],
                                     sum[m_idx][n_idx][3],
                                     reg_a[m_idx][0],
                                     reg_a[m_idx][1],
                                     reg_a[m_idx][2],
                                     reg_a[m_idx][3],
                                     reg_b[n_idx][0],
                                     reg_b[n_idx][1]);
                    } else {
                        M16N8K16_BF16(sum[m_idx][n_idx][0],
                                      sum[m_idx][n_idx][1],
                                      sum[m_idx][n_idx][2],
                                      sum[m_idx][n_idx][3],
                                      reg_a[m_idx][0],
                                      reg_a[m_idx][1],
                                      reg_a[m_idx][2],
                                      reg_a[m_idx][3],
                                      reg_b[n_idx][0],
                                      reg_b[n_idx][1]);
                    }
                }
            }
        }
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();
        read_idx ^= 1;
        write_idx ^= 1;
    }
    // ------------------- 流水线收尾：完成最后一组 MMA，再写回 C。
#pragma unroll
    for (int k_step = 0; k_step < 2; ++k_step) {
        int k_offset = k_step * 16;

        uint32_t reg_a[4][4];
        uint32_t reg_b[4][2];

        // 为 A 发出四次 ldmatrix 加载，覆盖 4×16=64 行。
#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
            // ldmatrix x4 的四个 8×8 子块在此共同组成一个 16×16 数据块。
            int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
            int a_col = k_offset + (lane_id / 16) * 8;
            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
            LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
        }

        // 为 B 发出四次 ldmatrix 加载，覆盖 4×8=32 列。
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            // 线程束内编号 0~15 的线程提供 16 行的起始地址，对应两个 8×8 子块。
            int b_row = k_offset + (lane_id % 16);
            int b_col = warp_id_n * 32 + n_idx * 8;

            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
            LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
        }

        // MMA 主体：在 M、N 方向各循环 4 次，执行 m16n8k16 矩阵乘加。
#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                if constexpr (std::is_same_v<T, __half>) {
                    M16N8K16_F16(sum[m_idx][n_idx][0],
                                 sum[m_idx][n_idx][1],
                                 sum[m_idx][n_idx][2],
                                 sum[m_idx][n_idx][3],
                                 reg_a[m_idx][0],
                                 reg_a[m_idx][1],
                                 reg_a[m_idx][2],
                                 reg_a[m_idx][3],
                                 reg_b[n_idx][0],
                                 reg_b[n_idx][1]);
                } else {
                    M16N8K16_BF16(sum[m_idx][n_idx][0],
                                  sum[m_idx][n_idx][1],
                                  sum[m_idx][n_idx][2],
                                  sum[m_idx][n_idx][3],
                                  reg_a[m_idx][0],
                                  reg_a[m_idx][1],
                                  reg_a[m_idx][2],
                                  reg_a[m_idx][3],
                                  reg_b[n_idx][0],
                                  reg_b[n_idx][1]);
                }
            }
        }
    }

    // ---------------- 写回 C ----------------
    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;
            if constexpr (std::is_same_v<T, __half>) {
                HALF2(c[(c_base_row + t_row) * n + c_base_col + t_col]) =
                    __float22half2_rn(FLOAT2(sum[m_idx][n_idx][0]));
                HALF2(c[(c_base_row + t_row + 8) * n + c_base_col + t_col]) =
                    __float22half2_rn(FLOAT2(sum[m_idx][n_idx][2]));
            } else {
                BFLOAT2(c[(c_base_row + t_row) * n + c_base_col + t_col]) =
                    __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][0]));
                BFLOAT2(c[(c_base_row + t_row + 8) * n + c_base_col + t_col]) =
                    __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][2]));
            }
        }
    }
}

// 一个线程块计算 C 的 128×128 数据块。
template <const int BM = 128, const int BN = 128, const int BK = 32, typename T>
__global__ void hgemm_bcf_dbf_rw_kernel(T *a, T *b, T *c, int m, int n, int k) {
    // 重排线程块到输出数据块的映射（网格 swizzle）。
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // 网格重排的逻辑分组宽度为 8。

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 全局内存到共享内存的加载索引。
    int load_a_row = tid / 4;        // 0~63
    int load_a_col = (tid % 4) * 8;  // 0,8,16,24
    int load_b_row = tid / 16;       // 0..15（K 维）
    int load_b_col = (tid % 16) * 8; // 0,8,...,120（N 维）

    // 用 union 复用同一片共享内存：先存 A/B 数据块，计算结束后改作 C 的写回缓冲。
    __shared__ __align__(128) union {
        // 第一阶段：A、B 数据块。
        struct {
            T As[2][BM][BK];
            T Bs[2][BK][BN];
        };
        // 第二阶段：C 的写回缓冲。
        T Cs[BM][BN];
    } smem;

    // 在线程块内部，继续按线程束划分输出数据块。
    // 每个线程束负责 C 的 64×32 数据块。
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 每线程的累加器：M 方向 4 个片段 × N 方向 4 个片段 × 每片段 4 个 float，共 64 个。
    float sum[4][4][4] = {0.f};

    // ----------------------------- 流水线序言：先预取一组 As/Bs。
    // 用 cp.async 异步加载 A。
    uint32_t smem_a0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&smem.As[0][load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
    uint32_t smem_a1 = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem.As[0][load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

    T *global_a0 = &a[(by * BM + load_a_row) * k + load_a_col];
    T *global_a1 = &a[(by * BM + load_a_row + 64) * k + load_a_col];

    CP_ASYNC_CG(smem_a0, global_a0);
    CP_ASYNC_CG(smem_a1, global_a1);

    // 用 cp.async 异步加载 B。
    uint32_t smem_b0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&smem.Bs[0][load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
    uint32_t smem_b1 = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem.Bs[0][load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));

    T *global_b0 = &b[(load_b_row)*n + bx * BN + load_b_col];
    T *global_b1 = &b[(load_b_row + 16) * n + bx * BN + load_b_col];

    CP_ASYNC_CG(smem_b0, global_b0);
    CP_ASYNC_CG(smem_b1, global_b1);

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP_0();
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    // 沿 K 维遍历数据块的主循环。
    for (int bk = 32; bk < k; bk += BK) {

        // 1. 用 cp.async 异步加载 A。
        smem_a0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.As[write_idx][load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
        smem_a1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.As[write_idx][load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

        // 按固定步长推进全局内存指针，减少反复构造地址所需的中间变量。
        global_a0 += BK;
        global_a1 += BK;

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        // 2. 用 cp.async 异步加载 B。
        smem_b0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.Bs[write_idx][load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
        smem_b1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.Bs[write_idx][load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));

        global_b0 += BK * n;
        global_b1 += BK * n;

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);

        CP_ASYNC_COMMIT_GROUP();

        // 3. Tensor Core 计算：K 维分两步，每步处理 16 个元素。
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 为 A 发出四次 ldmatrix 加载，覆盖 4×16=64 行。
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                // ldmatrix x4 的四个 8×8 子块在此共同组成一个 16×16 数据块。
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 8;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&smem.As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // 为 B 发出四次 ldmatrix 加载，覆盖 4×8=32 列。
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // 线程束内编号 0~15 的线程提供 16 行的起始地址，对应两个 8×8 子块。
                int b_row = k_offset + (lane_id % 16);
                int b_col = warp_id_n * 32 + n_idx * 8;

                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&smem.Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
                LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA 主体：在 M、N 方向各循环 4 次，执行 m16n8k16 矩阵乘加。
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    if constexpr (std::is_same_v<T, __half>) {
                        M16N8K16_F16(sum[m_idx][n_idx][0],
                                     sum[m_idx][n_idx][1],
                                     sum[m_idx][n_idx][2],
                                     sum[m_idx][n_idx][3],
                                     reg_a[m_idx][0],
                                     reg_a[m_idx][1],
                                     reg_a[m_idx][2],
                                     reg_a[m_idx][3],
                                     reg_b[n_idx][0],
                                     reg_b[n_idx][1]);
                    } else {
                        M16N8K16_BF16(sum[m_idx][n_idx][0],
                                      sum[m_idx][n_idx][1],
                                      sum[m_idx][n_idx][2],
                                      sum[m_idx][n_idx][3],
                                      reg_a[m_idx][0],
                                      reg_a[m_idx][1],
                                      reg_a[m_idx][2],
                                      reg_a[m_idx][3],
                                      reg_b[n_idx][0],
                                      reg_b[n_idx][1]);
                    }
                }
            }
        }
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();
        read_idx ^= 1;
        write_idx ^= 1;
    }
    // ------------------- 流水线收尾：完成最后一组 MMA，再写回 C。
#pragma unroll
    for (int k_step = 0; k_step < 2; ++k_step) {
        int k_offset = k_step * 16;

        uint32_t reg_a[4][4];
        uint32_t reg_b[4][2];

        // 为 A 发出四次 ldmatrix 加载，覆盖 4×16=64 行。
#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
            // ldmatrix x4 的四个 8×8 子块在此共同组成一个 16×16 数据块。
            int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
            int a_col = k_offset + (lane_id / 16) * 8;
            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&smem.As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
            LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
        }

        // 为 B 发出四次 ldmatrix 加载，覆盖 4×8=32 列。
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            // 线程束内编号 0~15 的线程提供 16 行的起始地址，对应两个 8×8 子块。
            int b_row = k_offset + (lane_id % 16);
            int b_col = warp_id_n * 32 + n_idx * 8;

            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&smem.Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
            LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
        }

        // MMA 主体：在 M、N 方向各循环 4 次，执行 m16n8k16 矩阵乘加。
#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                if constexpr (std::is_same_v<T, __half>) {
                    M16N8K16_F16(sum[m_idx][n_idx][0],
                                 sum[m_idx][n_idx][1],
                                 sum[m_idx][n_idx][2],
                                 sum[m_idx][n_idx][3],
                                 reg_a[m_idx][0],
                                 reg_a[m_idx][1],
                                 reg_a[m_idx][2],
                                 reg_a[m_idx][3],
                                 reg_b[n_idx][0],
                                 reg_b[n_idx][1]);
                } else {
                    M16N8K16_BF16(sum[m_idx][n_idx][0],
                                  sum[m_idx][n_idx][1],
                                  sum[m_idx][n_idx][2],
                                  sum[m_idx][n_idx][3],
                                  reg_a[m_idx][0],
                                  reg_a[m_idx][1],
                                  reg_a[m_idx][2],
                                  reg_a[m_idx][3],
                                  reg_b[n_idx][0],
                                  reg_b[n_idx][1]);
                }
            }
        }
    }

    // ---------------- 写回 C ----------------
    // 复用原先的 As/Bs 共享内存，暂存 C，便于后续合并写回。
    __syncthreads();

    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

    // 先把寄存器中的结果写到共享内存 Cs。
#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = warp_id_m * 64 + m_idx * 16; // M 方向：每个片段跨 16 行。
            int c_base_col = warp_id_n * 32 + n_idx * 8;  // N 方向：每个片段跨 8 列。

            // 每个片段分成两组 8 行，合计写出 16 行。
            int c_row_0 = c_base_row + t_row;
            int c_row_2 = c_base_row + t_row + 8;
            int c_col = c_base_col + t_col;

            if constexpr (std::is_same_v<T, __half>) {
                HALF2(smem.Cs[c_row_0][SWIZZLE_C(c_row_0, c_col)]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][0]));
                HALF2(smem.Cs[c_row_2][SWIZZLE_C(c_row_2, c_col)]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][2]));
            } else {
                BFLOAT2(smem.Cs[c_row_0][SWIZZLE_C(c_row_0, c_col)]) =
                    __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][0]));
                BFLOAT2(smem.Cs[c_row_2][SWIZZLE_C(c_row_2, c_col)]) =
                    __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][2]));
            }
        }
    }

    __syncthreads();

    // 再从共享内存搬到全局内存。
    // 每线程共搬 64 个 FP16/BF16 元素，即 8 次 float4；每一步每线程束搬 512 字节、每块搬 4096 字节。
    T *c_block = &c[by * BM * n + bx * BN];

#pragma unroll
    for (int step = 0; step < 8; ++step) {
        // 让线程束内相邻线程的 elem_idx 相邻，以形成连续的合并访存。
        int elem_idx = (step * 256 + tid) * 8;
        int row = elem_idx / 128;
        int col = elem_idx % 128;

        int s_col = SWIZZLE_C(row, col);

        FLOAT4(c_block[row * n + col]) = FLOAT4(smem.Cs[row][s_col]);
    }
}

// 这里只检查 CUDA 与连续性；没有完整验证设备一致、形状、dtype、尾部及对齐。
// 下面的宏负责在主机端取指针、计算网格并使用 PyTorch 当前流；启动通常是异步的。
#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

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
        if (a.dtype() == torch::kHalf) {                                                                               \
            name##_kernel<128, 128, 32>                                                                                \
                <<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<__half *>(a.data_ptr()),          \
                                                                    reinterpret_cast<__half *>(b.data_ptr()),          \
                                                                    reinterpret_cast<__half *>(c.data_ptr()),          \
                                                                    M,                                                 \
                                                                    N,                                                 \
                                                                    K);                                                \
        } else {                                                                                                       \
            name##_kernel<128, 128, 32>                                                                                \
                <<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<__nv_bfloat16 *>(a.data_ptr()),   \
                                                                    reinterpret_cast<__nv_bfloat16 *>(b.data_ptr()),   \
                                                                    reinterpret_cast<__nv_bfloat16 *>(c.data_ptr()),   \
                                                                    M,                                                 \
                                                                    N,                                                 \
                                                                    K);                                                \
        }                                                                                                              \
    }

binding_tiled_func_gen(hgemm_naive);
binding_tiled_func_gen(hgemm_bcf);
binding_tiled_func_gen(hgemm_bcf_dbf);
binding_tiled_func_gen(hgemm_bcf_dbf_rw);

extern void hgemm_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c);

// 导出供 Python 调用的绑定。

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(hgemm_cublas);
    torch_pybinding_func(hgemm_naive);
    torch_pybinding_func(hgemm_bcf);
    torch_pybinding_func(hgemm_bcf_dbf);
    torch_pybinding_func(hgemm_bcf_dbf_rw);
}
