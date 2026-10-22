/*
 * 中文导读：以 FP32 张量保存输入输出，用 TF32 的 m16n8k8 MMA 做矩阵乘加。
 * 32 个线程共同持有一个 MMA 片段；单个线程中的 4 个 float 累加器只代表该片段的一部分。
 * 线程块覆盖 128×128 输出，K 方向每次处理 16 个值；共享内存分块没有任意尾部保护。
 * 带网格重排的版本额外按 8 个列块分组，入门采用 N 为 1024 的倍数；各版本差异见 readme.md。
 * ldmatrix 的 b16 描述搬运粒度，不等于把输入 float 做了 FP16 数值转换。
 * 原路径没有显式的 FP32→TF32 数值转换；ldmatrix 和位重解释本身不完成该格式转换。
 * TF32 格式约定、数值误差与目标架构行为须单独核实；本次保留算法，不能视为完整 FP32 精度。
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

#define SWIZZLE_A(row, col) ((col) ^ (((row >> 1) & 0x3) << 2))

#define SWIZZLE_B(row, col) ((col) ^ (((((((row) >> 1)) ^ ((row) >> 3))) & 0x3) << 2))

#define SWIZZLE_B_F2(row, col) ((col) ^ (((row) & 0x7) << 3))

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

// mma.sync：线程束协作执行 TF32 矩阵乘加，以 FP32 累加结果。
#define M16N8K8(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1)                                                                \
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "                                                 \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                                         \
                 : "=f"(C0), "=f"(C1), "=f"(C2), "=f"(C3)                                                              \
                 : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), "f"(C0), "f"(C1), "f"(C2), "f"(C3))


// ------------------------------------------ 转置 B 后通过 ldmatrix 加载 ------------------------------------------

// 一个线程块计算 C 的 128×128 数据块。
template <const int BM = 128, const int BN = 128, const int BK = 16>
__global__ __launch_bounds__(256, 2) void sgemm_tf32_bt_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 数据加载的线程映射。
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12
    int load_b_row = tid / WARP_SIZE;       // 0~7（K 维）
    int load_b_col = (tid % WARP_SIZE) * 4; // 0~124（N 维）

    // A 保持行主序，B 在共享内存中转置为列主序。
    __shared__ float As[BM][BK];
    __shared__ float Bs[BN][BK];

    // 线程束按 2×4 分块，每个线程束计算 C 的一个 64×32 数据块。
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 每线程的累加器共 4 个 M 片段 × 4 个 N 片段 × 每片段 4 个寄存器，即 64 个 float。
    float sum[4][4][4] = {0.f};

    // 主循环。
    for (int bk = 0; bk < k; bk += BK) {

        // 1. 用 cp.async 加载 A，访问地址按 16 字节对齐。
        uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][load_a_col]));
        uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row + 64][load_a_col]));

        float *global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        float *global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);
        CP_ASYNC_COMMIT_GROUP();

        // 2. 加载 B，并手动转置后写入共享内存。
        float4 tmp_b0 = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        float4 tmp_b1 = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        // 将 B 手动转置后存入共享内存。
        Bs[load_b_col + 0][load_b_row] = tmp_b0.x;
        Bs[load_b_col + 1][load_b_row] = tmp_b0.y;
        Bs[load_b_col + 2][load_b_row] = tmp_b0.z;
        Bs[load_b_col + 3][load_b_row] = tmp_b0.w;
        Bs[load_b_col + 0][load_b_row + 8] = tmp_b1.x;
        Bs[load_b_col + 1][load_b_row + 8] = tmp_b1.y;
        Bs[load_b_col + 2][load_b_row + 8] = tmp_b1.z;
        Bs[load_b_col + 3][load_b_row + 8] = tmp_b1.w;

        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // 3. Tensor Core 计算：K 维分两步，每步处理 8 个元素。
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 为 A 的片段发出四次 ldmatrix 加载，覆盖 4×16=64 行。
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                // warp_id_m 对应的行跨度为 64。
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16); // 0~15, 0~15
                int a_col = k_offset + (lane_id / 16) * 4;                // 0,4  8,12
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][a_col]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // 为 B 的片段发出四次 ldmatrix 加载，覆盖 4×8=32 列。
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // warp_id_n 对应的列跨度为 32。
                int b_row = warp_id_n * 32 + n_idx * 8 + (lane_id % 8);
                int b_col = k_offset + ((lane_id / 8) % 2) * 4;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][b_col]));
                LDMATRIX_X2(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA 核心：在 M、N 方向各循环 4 次，执行 m16n8k8 矩阵乘加。
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    M16N8K8(sum[m_idx][n_idx][0],
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
        __syncthreads();
    }

    // m16n8k8 的 C 片段布局映射。
    // C 片段在各线程寄存器中的布局见下方 PTX 文档。
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688 1688.tf32
    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {

            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            FLOAT2(c[(c_base_row + t_row) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][0]);
            FLOAT2(c[(c_base_row + t_row + 8) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][2]);
        }
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 16>
__global__ __launch_bounds__(256,
                             2) void sgemm_tf32_bt_swizzle_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 数据加载的线程映射。
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12
    int load_b_row = tid / WARP_SIZE;       // 0~7（K 维）
    int load_b_col = (tid % WARP_SIZE) * 4; // 0~124（N 维）

    // A 保持行主序，B 在共享内存中转置为列主序。
    __shared__ float As[BM][BK];
    __shared__ float Bs[BN][BK];

    // 线程束分块：按行主序映射，便于合并 C 的全局内存写回。
    // 每个线程束计算 C 的一个 64×32 数据块。
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 每线程的累加器共 4 个 M 片段 × 4 个 N 片段 × 每片段 4 个寄存器，即 64 个 float。
    float sum[4][4][4] = {0.f};

    // 主循环。
    for (int bk = 0; bk < k; bk += BK) {

        // 1. 用 cp.async 加载 A，并在共享内存中应用 swizzle 重排。
        uint32_t smem_a0 =
            static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
        uint32_t smem_a1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

        float *global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        float *global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);
        CP_ASYNC_COMMIT_GROUP();

        // 2. 加载 B。
        float4 tmp_b0 = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        float4 tmp_b1 = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        // 将 B 转置并按 SWIZZLE_B 分散写入共享内存。
        Bs[load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row)] = tmp_b0.x;
        Bs[load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row)] = tmp_b0.y;
        Bs[load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row)] = tmp_b0.z;
        Bs[load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row)] = tmp_b0.w;

        Bs[load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row + 8)] = tmp_b1.x;
        Bs[load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row + 8)] = tmp_b1.y;
        Bs[load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row + 8)] = tmp_b1.z;
        Bs[load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row + 8)] = tmp_b1.w;

        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // 3. Tensor Core 计算。
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 为 A 的片段发出四次 ldmatrix 加载，使用与写入时一致的 swizzle 地址映射。
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // 为 B 的片段发出四次 ldmatrix 加载，使用与写入时一致的 swizzle 地址映射。
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                int b_row = warp_id_n * 32 + n_idx * 8 + (lane_id % 8);
                int b_col = k_offset + ((lane_id / 8) % 2) * 4;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][SWIZZLE_B(b_row, b_col)]));
                LDMATRIX_X2(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA 核心：在 M、N 方向各循环 4 次，执行 m16n8k8 矩阵乘加。
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    M16N8K8(sum[m_idx][n_idx][0],
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
        __syncthreads();
    }

    // ---------------- 写回 C ----------------
    int t_row = lane_id / 4;
    int t_col = (lane_id % 4) * 2;

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            FLOAT2(c[(c_base_row + t_row) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][0]);
            FLOAT2(c[(c_base_row + t_row + 8) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][2]);
        }
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 16>
__global__
__launch_bounds__(256, 2) void sgemm_tf32_bt_swizzle_dbf_kernel(float *a, float *b, float *c, int m, int n, int k) {
    // 重排线程块到输出数据块的映射（网格 swizzle）。
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // 网格重排的分组宽度。

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;
    // int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 4;
    int load_b_row = tid / WARP_SIZE;
    int load_b_col = (tid % WARP_SIZE) * 4;

    // 双缓冲。
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BN][BK];

    // 线程束按 2×4 分块。
    int warp_id_m = warp_id / 4;
    int warp_id_n = warp_id % 4;

    float sum[4][4][4] = {0.f};
    // ------------------------------------------- 预取第一个数据块。
    int a_swizzle_col_0 = SWIZZLE_A(load_a_row, load_a_col);
    int a_swizzle_col_1 = SWIZZLE_A(load_a_row + 64, load_a_col);

    uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[0][load_a_row][a_swizzle_col_0]));
    uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[0][load_a_row + 64][a_swizzle_col_1]));

    float *global_a0 = &a[(by * BM + load_a_row) * k + 0 + load_a_col];
    float *global_a1 = &a[(by * BM + load_a_row + 64) * k + 0 + load_a_col];

    CP_ASYNC_CG(smem_a0, global_a0);
    CP_ASYNC_CG(smem_a1, global_a1);
    CP_ASYNC_COMMIT_GROUP();

    // 先把 B 同步加载到寄存器。
    float4 tmp_b0 = FLOAT4(b[(0 + load_b_row) * n + bx * BN + load_b_col]);
    float4 tmp_b1 = FLOAT4(b[(0 + load_b_row + 8) * n + bx * BN + load_b_col]);

    // 将 B 转置后存入共享内存。
    Bs[0][load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row)] = tmp_b0.x;
    Bs[0][load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row)] = tmp_b0.y;
    Bs[0][load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row)] = tmp_b0.z;
    Bs[0][load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row)] = tmp_b0.w;

    Bs[0][load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row + 8)] = tmp_b1.x;
    Bs[0][load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row + 8)] = tmp_b1.y;
    Bs[0][load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row + 8)] = tmp_b1.z;
    Bs[0][load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row + 8)] = tmp_b1.w;

    CP_ASYNC_WAIT_GROUP_0();
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    // 主循环。
    for (int bk = BK; bk < k; bk += BK) {

        // 1. 将下一个 A 数据块异步预取到 write_idx 指定的缓冲。
        smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[write_idx][load_a_row][a_swizzle_col_0]));
        smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[write_idx][load_a_row + 64][a_swizzle_col_1]));
        global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        // 把下一个 B 数据块预取到寄存器，尝试用当前计算覆盖全局内存加载延迟。
        tmp_b0 = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        tmp_b1 = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);
        CP_ASYNC_COMMIT_GROUP();

        // 2. 使用 read_idx 指定的当前缓冲进行 Tensor Core 计算。
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;
            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                int b_row = warp_id_n * 32 + n_idx * 8 + (lane_id % 8);
                int b_col = k_offset + ((lane_id / 8) % 2) * 4;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
                LDMATRIX_X2(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    M16N8K8(sum[m_idx][n_idx][0],
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

        // 3. 将预取到寄存器的 B 写入共享内存的写缓冲。
        Bs[write_idx][load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row)] = tmp_b0.x;
        Bs[write_idx][load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row)] = tmp_b0.y;
        Bs[write_idx][load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row)] = tmp_b0.z;
        Bs[write_idx][load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row)] = tmp_b0.w;

        Bs[write_idx][load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row + 8)] = tmp_b1.x;
        Bs[write_idx][load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row + 8)] = tmp_b1.y;
        Bs[write_idx][load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row + 8)] = tmp_b1.z;
        Bs[write_idx][load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row + 8)] = tmp_b1.w;

        // 4. 等待异步拷贝并进行线程块同步。
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // 交换读、写缓冲。
        read_idx ^= 1;
        write_idx ^= 1;
    }
    // 处理最后一个预取的数据块。
#pragma unroll
    for (int k_step = 0; k_step < 2; ++k_step) {
        int k_offset = k_step * 8;
        uint32_t reg_a[4][4];
        uint32_t reg_b[4][2];

#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
            int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
            int a_col = k_offset + (lane_id / 16) * 4;
            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
            LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
        }

#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int b_row = warp_id_n * 32 + n_idx * 8 + (lane_id % 8);
            int b_col = k_offset + ((lane_id / 8) % 2) * 4;
            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
            LDMATRIX_X2(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
        }

#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                M16N8K8(sum[m_idx][n_idx][0],
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

    // ---------------- 写回 C ----------------
    int t_row = lane_id / 4;
    int t_col = (lane_id % 4) * 2;

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            FLOAT2(c[(c_base_row + t_row) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][0]);
            FLOAT2(c[(c_base_row + t_row + 8) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][2]);
        }
    }
}

// ------------------------------------ 用 cp.async 加载 A/B，再从共享内存加载 Bs ------------------------------------

// 一个线程块计算 C 的 128×128 数据块。
template <const int BM = 128, const int BN = 128, const int BK = 16>
__global__ __launch_bounds__(256,
                             2) void sgemm_tf32_swizzle_bcf_kernel(float *a, float *b, float *c, int m, int n, int k) {
    // 重排线程块到输出数据块的映射（网格 swizzle）。
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // 网格重排的分组宽度。

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;
    // int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 数据加载的线程映射。
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12
    int load_b_row = tid / WARP_SIZE;       // 0~7（K 维）
    int load_b_col = (tid % WARP_SIZE) * 4; // 0~124（N 维）

    // A/B 都按行主序保存；此处 A 为 [BM][BK]，B 为 [BK][BN]，以声明为准。
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 线程束分块：每个线程束计算 C 的一个 64×32 数据块。
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 每线程的累加器共 4 个 M 片段 × 4 个 N 片段 × 每片段 4 个寄存器，即 64 个 float。
    float sum[4][4][4] = {0.f};

    // 主循环。
    for (int bk = 0; bk < k; bk += BK) {

        // 1. 用 cp.async 异步加载 A。
        uint32_t smem_a0 =
            static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
        uint32_t smem_a1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

        float *global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        float *global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        // 用 cp.async 异步加载 B。
        uint32_t smem_b0 =
            static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_row][SWIZZLE_B_F2(load_b_row, load_b_col)]));
        uint32_t smem_b1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[load_b_row + 8][SWIZZLE_B_F2(load_b_row + 8, load_b_col)]));

        float *global_b0 = &b[(bk + load_b_row) * n + bx * BN + load_b_col];
        float *global_b1 = &b[(bk + load_b_row + 8) * n + bx * BN + load_b_col];

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);
        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // 3. Tensor Core 计算：K 维分两步，每步处理 8 个元素。
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 为 A 的片段发出四次 ldmatrix 加载，覆盖 4×16=64 行。
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // B 片段：每线程持有相隔 4 行的两个值；每 4 个线程合计覆盖 8 行×1 列。
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                int n_base = warp_id_n * 32 + n_idx * 8;
                int b_col = n_base + (lane_id / 4);
                int b_row_0 = k_offset + (lane_id % 4);
                int b_row_1 = k_offset + (lane_id % 4) + 4;

                reg_b[n_idx][0] = __float_as_uint(Bs[b_row_0][SWIZZLE_B_F2(b_row_0, b_col)]);
                reg_b[n_idx][1] = __float_as_uint(Bs[b_row_1][SWIZZLE_B_F2(b_row_1, b_col)]);
            }

            // MMA 核心：在 M、N 方向各循环 4 次，执行 m16n8k8 矩阵乘加。
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    M16N8K8(sum[m_idx][n_idx][0],
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

            FLOAT2(c[(c_base_row + t_row) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][0]);
            FLOAT2(c[(c_base_row + t_row + 8) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][2]);
        }
    }
}

// 一个线程块计算 C 的 128×128 数据块。
template <const int BM = 128, const int BN = 128, const int BK = 16>
__global__ void sgemm_tf32_swizzle_bcf_dbf_kernel(float *a, float *b, float *c, int m, int n, int k) {
    // 重排线程块到输出数据块的映射（网格 swizzle）。
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // 网格重排的分组宽度。

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    // int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 4;
    int load_b_row = tid / WARP_SIZE;
    int load_b_col = (tid % WARP_SIZE) * 4;

    // 双缓冲。
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BM];

    // 线程束按 2×4 分块。
    int warp_id_m = warp_id / 4;
    int warp_id_n = warp_id % 4;

    float sum[4][4][4] = {0.f};
    // ------------------------------------------- 预取第一个数据块。
    int a_swizzle_col_0 = SWIZZLE_A(load_a_row, load_a_col);
    int a_swizzle_col_1 = SWIZZLE_A(load_a_row + 64, load_a_col);

    uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[0][load_a_row][a_swizzle_col_0]));
    uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[0][load_a_row + 64][a_swizzle_col_1]));

    float *global_a0 = &a[(by * BM + load_a_row) * k + 0 + load_a_col];
    float *global_a1 = &a[(by * BM + load_a_row + 64) * k + 0 + load_a_col];
    CP_ASYNC_CG(smem_a0, global_a0);
    CP_ASYNC_CG(smem_a1, global_a1);

    uint32_t smem_b0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[0][load_b_row][SWIZZLE_B_F2(load_b_row, load_b_col)]));
    uint32_t smem_b1 = static_cast<uint32_t>(
        __cvta_generic_to_shared(&Bs[0][load_b_row + 8][SWIZZLE_B_F2(load_b_row + 8, load_b_col)]));

    float *global_b0 = &b[(0 + load_b_row) * n + bx * BN + load_b_col];
    float *global_b1 = &b[(0 + load_b_row + 8) * n + bx * BN + load_b_col];

    CP_ASYNC_CG(smem_b0, global_b0);
    CP_ASYNC_CG(smem_b1, global_b1);

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP_0();
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    // 主循环。
    for (int bk = BK; bk < k; bk += BK) {

        // 1. 将下一个 A 数据块异步预取到 write_idx 指定的缓冲。
        smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[write_idx][load_a_row][a_swizzle_col_0]));
        smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[write_idx][load_a_row + 64][a_swizzle_col_1]));
        global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];
        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        // 用 cp.async 异步加载 B。
        uint32_t smem_b0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[write_idx][load_b_row][SWIZZLE_B_F2(load_b_row, load_b_col)]));
        uint32_t smem_b1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[write_idx][load_b_row + 8][SWIZZLE_B_F2(load_b_row + 8, load_b_col)]));

        float *global_b0 = &b[(bk + load_b_row) * n + bx * BN + load_b_col];
        float *global_b1 = &b[(bk + load_b_row + 8) * n + bx * BN + load_b_col];

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);
        CP_ASYNC_COMMIT_GROUP();

        // 2. 使用 read_idx 指定的当前缓冲进行 Tensor Core 计算。
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;
            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // M 维覆盖 4×16=64 行。
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                int n_base = warp_id_n * 32 + n_idx * 8;
                int b_col = n_base + (lane_id / 4);
                int b_row_0 = k_offset + (lane_id % 4);
                int b_row_1 = k_offset + (lane_id % 4) + 4;

                reg_b[n_idx][0] = __float_as_uint(Bs[read_idx][b_row_0][SWIZZLE_B_F2(b_row_0, b_col)]);
                reg_b[n_idx][1] = __float_as_uint(Bs[read_idx][b_row_1][SWIZZLE_B_F2(b_row_1, b_col)]);
            }

#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    M16N8K8(sum[m_idx][n_idx][0],
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

        // 3. 等待 cp.async 拷贝，并进行线程块同步。
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // 交换读、写缓冲。
        read_idx ^= 1;
        write_idx ^= 1;
    }
    // 处理最后一个预取的数据块。
#pragma unroll
    for (int k_step = 0; k_step < 2; ++k_step) {
        int k_offset = k_step * 8;
        uint32_t reg_a[4][4];
        uint32_t reg_b[4][2];

#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
            int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
            int a_col = k_offset + (lane_id / 16) * 4;
            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
            LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
        }

#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int n_base = warp_id_n * 32 + n_idx * 8;
            int b_col = n_base + (lane_id / 4);
            int b_row_0 = k_offset + (lane_id % 4);
            int b_row_1 = k_offset + (lane_id % 4) + 4;

            reg_b[n_idx][0] = __float_as_uint(Bs[read_idx][b_row_0][SWIZZLE_B_F2(b_row_0, b_col)]);
            reg_b[n_idx][1] = __float_as_uint(Bs[read_idx][b_row_1][SWIZZLE_B_F2(b_row_1, b_col)]);
        }

#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                M16N8K8(sum[m_idx][n_idx][0],
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

    // ---------------- 写回 C ----------------
    int t_row = lane_id / 4;
    int t_col = (lane_id % 4) * 2;

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            FLOAT2(c[(c_base_row + t_row) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][0]);
            FLOAT2(c[(c_base_row + t_row + 8) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][2]);
        }
    }
    // 尝试复用 As/Bs 来合并 C 的写回；下方是保留的未启用代码，布局处理较复杂。
    // 实验代码保持注释状态，仅翻译其中的坐标说明。
    /**
    float (*Cs)[128] = (float (*)[128])(&As[0][0][0]);

    #define SWIZZLE_C(row, col) ((col) ^ (((row) << 3) & 127))

    int t_row = lane_id / 4;
    int t_col = (lane_id % 4) * 2;

    for (int m_idx = 0; m_idx < 4; ++m_idx) {
        __syncthreads();

#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int smem_row = warp_id_m * 16 + t_row;
            int smem_col = warp_id_n * 32 + n_idx * 8 + t_col;

            FLOAT2(Cs[smem_row][SWIZZLE_C(smem_row, smem_col)]) = FLOAT2(sum[m_idx][n_idx][0]);
            FLOAT2(Cs[smem_row + 8][SWIZZLE_C(smem_row + 8, smem_col)]) = FLOAT2(sum[m_idx][n_idx][2]);
        }

        __syncthreads();

        int t_c_row = tid / 32;
        int t_c_col = (tid % 32) * 4;

#pragma unroll
        for (int step = 0; step < 4; ++step) {
            int smem_row = t_c_row + step * 8;
            int smem_col = t_c_col;

            float4 res = FLOAT4(Cs[smem_row][SWIZZLE_C(smem_row, smem_col)]);

            // 恢复原矩阵中的实际行列坐标
            int global_row = by * BM + m_idx * 16 + (smem_row < 16 ? smem_row : 64 + smem_row - 16);
            int global_col = bx * BN + smem_col;

            FLOAT4(c[global_row * n + global_col]) = res;
        }
    }*/
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
        name##_kernel<128, 128, 16><<<blocks_per_grid, threads_per_block, 0, stream>>>(                                \
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);                                   \
    }

binding_tiled_func_gen(sgemm_tf32_bt);
binding_tiled_func_gen(sgemm_tf32_bt_swizzle);
binding_tiled_func_gen(sgemm_tf32_bt_swizzle_dbf);
binding_tiled_func_gen(sgemm_tf32_swizzle_bcf);
binding_tiled_func_gen(sgemm_tf32_swizzle_bcf_dbf);
