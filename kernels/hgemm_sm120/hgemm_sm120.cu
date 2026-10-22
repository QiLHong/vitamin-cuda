/*
 * 中文导读：面向 SM120 的 FP16/BF16 GEMM，比较 cp.async 与 TMA 数据搬运流水线。
 * 每块 256 线程计算 128×128 输出；MMA 以 FP32 累加，最后转换回 FP16/BF16。
 * TMA 描述符提供全局形状/步长，mbarrier 追踪各阶段的到达与传输完成；二者不可混为一个计数。
 * 沿 K 维的共享内存阶段与阶段内的寄存器双缓冲是两层不同的流水线。
 * 当前包装未完整检查输入关系，非 FP16 分支直接按 BF16 解释；也没有通用尾部路径。
 * 128×128 输出分块和宽度为 8 的网格重排要求关注 M/N 边界；各版本 K 的预取限制见 readme.md。
 * 架构、TMA 对齐、异步代理可见性与屏障复用须在目标 GPU 验证；本次只改注释。
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

#define SWIZZLE_64B_TMA(row, col) ((col) ^ (((row >> 1) & 0x3) << 3))

#define SWIZZLE_128B_TMA(row, col) ((col) ^ (((row) & 0x7) << 3))

// ---------------- 内联 PTX 汇编宏 ----------------
// cp.async：将全局内存 src 中的 16 字节异步拷贝到共享内存 dst_smem_32b。
#define CP_ASYNC_CG(dst_smem_32b, src_global_ptr)                                                                      \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst_smem_32b), "l"(src_global_ptr))

// commit_group 将本线程此前的异步拷贝组成一组；wait_group 0 等待这些组完成。
// wait_group 不代替线程块屏障，后续跨线程使用共享内存时仍须协调。
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    if constexpr (N == 0)
        asm volatile("cp.async.wait_group 0;\n" ::);
    else if constexpr (N == 1)
        asm volatile("cp.async.wait_group 1;\n" ::);
    else if constexpr (N == 2)
        asm volatile("cp.async.wait_group 2;\n" ::);
    else if constexpr (N == 3)
        asm volatile("cp.async.wait_group 3;\n" ::);
}

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

// ---------------- MBarrier 与 TMA 的 PTX 辅助函数 ----------------

// MBarrier 对象：占 8 字节，使用时须满足 8 字节对齐。
typedef uint64_t mbarrier_t;

// 初始化屏障，由流水线序言中的单个线程调用。
__device__ __forceinline__ void mbarrier_init(mbarrier_t *mbar, uint32_t expected_count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))),
                 "r"(expected_count));
}

// 登记本阶段预期完成的 TMA 传输字节数，同时执行一次 arrive。
__device__ __forceinline__ void mbarrier_expect_tx(mbarrier_t *mbar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(mbar))),
                 "r"(tx_bytes));
}

// 消费者读完共享内存后可调用 arrive；此辅助函数在本文件没有被调用。
// 本实现没有专用生产者线程束：tid=0 发起 TMA，所有线程束参与计算；阶段复用另用 __syncthreads() 协调。
__device__ __forceinline__ void mbarrier_arrive(mbarrier_t *mbar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))));
}

// 等待指定屏障阶段完成，用于确认对应的 TMA 拷贝已结束。
__device__ __forceinline__ void mbarrier_wait(uint64_t *smem_ptr, uint32_t phase) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    // 单次 try_wait 未完成会再次循环，不是整个核函数的超时上限。
    // 依据 NVIDIA PTX 的 mbarrier.try_wait 定义，详见本目录 readme.md 中的官方链接。
    // 等待时间提示为 0x989680=10,000,000 纳秒；原注释写 cycles，按 PTX 规范应为纳秒。
    uint32_t ticks = 0x989680;

    asm volatile("{\n\t"
                 ".reg .pred p; \n\t"
                 "LAB_WAIT: \n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2; \n\t"
                 "@p bra DONE; \n\t"
                 "bra LAB_WAIT; \n\t"
                 "DONE: \n\t"
                 "}\n"
                 :
                 : "r"(smem_addr), "r"(phase), "r"(ticks)
                 : "memory");
}

// 线程块内使用的二维 TMA：全局内存 -> 共享内存。
__device__ __forceinline__ void cp_async_bulk_tensor_2d(
    mbarrier_t *mbar, const void *tmap, const void *smem_ptr, int32_t fast_coord, int32_t slow_coord) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));

    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3}], [%4];\n" ::"r"(smem_addr),
                 "l"(tmap),
                 "r"(fast_coord),
                 "r"(slow_coord),
                 "r"(mbar_addr)
                 : "memory");
}

template <const int BK, typename T>
__device__ __forceinline__ void
ldmatrix_A(uint32_t reg_a[4][4], T (*As)[BK], int warp_id_m, int lane_id, int k_offset) {

    // 为 A 发出四次 ldmatrix 加载，覆盖 4×16=64 行。
#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
        // ldmatrix x4 的四个 8×8 子块在此共同组成一个 16×16 数据块。
        int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
        int a_col = k_offset + (lane_id / 16) * 8;
        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_A(a_row, a_col)]));
        LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
    }
}

template <const int BK, typename T>
__device__ __forceinline__ void
ldmatrix_A_tma(uint32_t reg_a[4][4], T (*As)[BK], int warp_id_m, int lane_id, int k_offset) {

    // 为 A 发出四次 ldmatrix 加载，覆盖 4×16=64 行。
#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
        int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
        int a_col = k_offset + (lane_id / 16) * 8;
        if constexpr (BK == 32) {
            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_64B_TMA(a_row, a_col)]));
            LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
        } else {
            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_128B_TMA(a_row, a_col)]));
            LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
        }
    }
}

template <const int BN, const int BK, typename T>
__device__ __forceinline__ void
ldmatrix_B(uint32_t reg_b[4][2], T (*Bs)[BN], int warp_id_n, int lane_id, int k_offset) {

    // 为 B 发出四次 ldmatrix 加载，覆盖 4×8=32 列。
#pragma unroll
    for (int n_idx = 0; n_idx < 4; ++n_idx) {
        // 线程束内编号 0~15 的线程提供 16 行的起始地址，对应两个 8×8 子块。
        int b_row = k_offset + (lane_id % 16);
        int b_col = warp_id_n * 32 + n_idx * 8;

        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][SWIZZLE_B(b_row, b_col)]));
        LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
    }
}

template <const int BN, const int BK, typename T>
__device__ __forceinline__ void
ldmatrix_B_tma(uint32_t reg_b[4][2], T (*Bs)[BK][BN / 2], int warp_id_n, int lane_id, int k_offset) {
#pragma unroll
    for (int n_idx = 0; n_idx < 4; ++n_idx) {
        int b_row = k_offset + (lane_id % 16);
        int b_col = warp_id_n * 32 + n_idx * 8;

        // 依据分片编号选择共享内存索引。
        int chunk_idx = b_col / (BN / 2);
        int local_col = b_col % (BN / 2);

        uint32_t smem_addr =
            static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[chunk_idx][b_row][SWIZZLE_128B_TMA(b_row, local_col)]));
        LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
    }
}

template <typename T>
__device__ __forceinline__ void mma_compute(float sum[4][4][4], uint32_t reg_a[4][4], uint32_t reg_b[4][2]) {

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

template <const int BK, typename T>
__device__ __forceinline__ void cp_async_load_A(T (*As)[BK], int load_a_row, int load_a_col, T *global_a_ptr, int k) {

    // 用 cp.async 加载 A：每线程搬两行中的片段，两行的行号相差 64。
    uint32_t smem_a0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
    uint32_t smem_a1 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));
    // 两次加载分别覆盖 A 的前 64 行和后 64 行。
    CP_ASYNC_CG(smem_a0, global_a_ptr);
    CP_ASYNC_CG(smem_a1, global_a_ptr + 64 * k);
}

template <const int BK, const int BN, typename T>
__device__ __forceinline__ void cp_async_load_B(T (*Bs)[BN], int load_b_row, int load_b_col, T *global_b_ptr, int n) {

    // 用 cp.async 加载 B：每线程搬两行中的片段，两行的行号相差 16。
    uint32_t smem_b0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
    uint32_t smem_b1 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));
    // 两次加载分别覆盖 B 的前 16 行和后 16 行。
    CP_ASYNC_CG(smem_b0, global_b_ptr);
    CP_ASYNC_CG(smem_b1, global_b_ptr + 16 * n);
}

template <const int BM, const int BN, typename T>
__device__ __forceinline__ void write_c_via_smem(
    T *c, int by, int bx, int n, float sum[4][4][4], int warp_id_m, int warp_id_n, int lane_id, int tid, T (*Cs)[BN]) {

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
                HALF2(Cs[c_row_0][SWIZZLE_C(c_row_0, c_col)]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][0]));
                HALF2(Cs[c_row_2][SWIZZLE_C(c_row_2, c_col)]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][2]));
            } else {
                BFLOAT2(Cs[c_row_0][SWIZZLE_C(c_row_0, c_col)]) = __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][0]));
                BFLOAT2(Cs[c_row_2][SWIZZLE_C(c_row_2, c_col)]) = __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][2]));
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

        FLOAT4(c_block[row * n + col]) = FLOAT4(Cs[row][s_col]);
    }
}

// ------------------------------------------ cp.async 异步拷贝与 MMA 计算 ------------------------------------------

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

    T *global_a_ptr = &a[(by * BM + load_a_row) * k + load_a_col];
    T *global_b_ptr = &b[load_b_row * n + bx * BN + load_b_col];

    // ----------------------------- 流水线序言：先预取一组 As/Bs。
    // 按步长加载，完整覆盖 128×32 的 A 数据块与 32×128 的 B 数据块。
    cp_async_load_A<BK>(smem.As[0], load_a_row, load_a_col, global_a_ptr, k);
    cp_async_load_B<BK, BN>(smem.Bs[0], load_b_row, load_b_col, global_b_ptr, n);

    CP_ASYNC_COMMIT_GROUP();
    cp_async_wait_group<0>();
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    // 沿 K 维遍历数据块的主循环。
    for (int bk = 32; bk < k; bk += BK) {

        // 推进源数据指针。
        global_a_ptr += BK;
        global_b_ptr += BK * n;

        // 1. 用 cp.async 异步加载 A/B。
        cp_async_load_A<BK>(smem.As[write_idx], load_a_row, load_a_col, global_a_ptr, k);
        cp_async_load_B<BK, BN>(smem.Bs[write_idx], load_b_row, load_b_col, global_b_ptr, n);

        CP_ASYNC_COMMIT_GROUP();

        // 2. Tensor Core 计算：K 维分两步，每步处理 16 个元素。
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 为 A 发出四次 ldmatrix 加载，覆盖 4×16=64 行。
            ldmatrix_A<BK>(reg_a, smem.As[read_idx], warp_id_m, lane_id, k_offset);

            // 为 B 发出四次 ldmatrix 加载，覆盖 4×8=32 列。
            ldmatrix_B<BN, BK>(reg_b, smem.Bs[read_idx], warp_id_n, lane_id, k_offset);

            // MMA 主体：在 M、N 方向各循环 4 次，执行 m16n8k16 矩阵乘加。
            mma_compute<T>(sum, reg_a, reg_b);
        }

        read_idx ^= 1;
        write_idx ^= 1;

        cp_async_wait_group<0>();
        __syncthreads();
    }
    // ------------------- 流水线收尾：完成最后一组 MMA，再写回 C。
#pragma unroll
    for (int k_step = 0; k_step < 2; ++k_step) {
        int k_offset = k_step * 16;

        uint32_t reg_a[4][4];
        uint32_t reg_b[4][2];

        // 为 A 发出四次 ldmatrix 加载，覆盖 4×16=64 行。
        ldmatrix_A<BK>(reg_a, smem.As[read_idx], warp_id_m, lane_id, k_offset);

        // 为 B 发出四次 ldmatrix 加载，覆盖 4×8=32 列。
        ldmatrix_B<BN, BK>(reg_b, smem.Bs[read_idx], warp_id_n, lane_id, k_offset);

        // MMA 主体：在 M、N 方向各循环 4 次，执行 m16n8k16 矩阵乘加。
        mma_compute<T>(sum, reg_a, reg_b);
    }

    write_c_via_smem<BM, BN>(c, by, bx, n, sum, warp_id_m, warp_id_n, lane_id, tid, smem.Cs);
}

template <const int BM = 128, const int BN = 128, const int BK = 32, const int STAGES = 3, typename T>
__global__ void hgemm_k_stages_kernel(T *a, T *b, T *c, int m, int n, int k) {
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
            T As[STAGES][BM][BK];
            T Bs[STAGES][BK][BN];
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

    T *global_a_ptr = &a[(by * BM + load_a_row) * k + load_a_col];
    T *global_b_ptr = &b[load_b_row * n + bx * BN + load_b_col];

    // 1. 流水线序言：预取 STAGES-1 组 As/Bs。
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
        cp_async_load_A<BK>(smem.As[i], load_a_row, load_a_col, global_a_ptr, k);
        cp_async_load_B<BK, BN>(smem.Bs[i], load_b_row, load_b_col, global_b_ptr, n);

        CP_ASYNC_COMMIT_GROUP();

        global_a_ptr += BK;
        global_b_ptr += BK * n;
    }
    // 当 STAGES=3 时，序言提交两组拷贝；等待最早一组时，较新的一组仍可在途。
    cp_async_wait_group<STAGES - 2>();
    __syncthreads();

    // 初始化流水线状态。
    int load_stage = STAGES - 1; // 下一次加载要填充的阶段。
    int compute_stage = 0;       // 当前供 MMA 读取的阶段。

    // 2. 主循环。
    for (int bk = (STAGES - 1) * BK; bk < k; bk += BK) {

        // 1. 向 load_stage 指定的缓冲发出 cp.async 加载。
        cp_async_load_A<BK>(smem.As[load_stage], load_a_row, load_a_col, global_a_ptr, k);
        cp_async_load_B<BK, BN>(smem.Bs[load_stage], load_b_row, load_b_col, global_b_ptr, n);

        CP_ASYNC_COMMIT_GROUP();

        // 2. Tensor Core 计算：K 维分两步，每步处理 16 个元素。
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;
            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 为 A 发出四次 ldmatrix 加载，覆盖 4×16=64 行。
            ldmatrix_A<BK>(reg_a, smem.As[compute_stage], warp_id_m, lane_id, k_offset);

            // 为 B 发出四次 ldmatrix 加载，覆盖 4×8=32 列。
            ldmatrix_B<BN, BK>(reg_b, smem.Bs[compute_stage], warp_id_n, lane_id, k_offset);

            // MMA 主体：在 M、N 方向各循环 4 次，执行 m16n8k16 矩阵乘加。
            mma_compute<T>(sum, reg_a, reg_b);
        }

        // 推进源数据指针。
        global_a_ptr += BK;
        global_b_ptr += BK * n;

        // 将流水线的读、写位置各推进一个阶段。
        load_stage = (load_stage + 1 == STAGES) ? 0 : load_stage + 1;
        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;

        // 等待最早尚未完成的拷贝组，随后再由线程块屏障协调共享内存访问。
        cp_async_wait_group<STAGES - 2>();
        __syncthreads();
    }

    // 3. 流水线收尾：处理剩余 STAGES-1 组数据，再写回结果。
    cp_async_wait_group<0>();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;
            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 为 A 发出四次 ldmatrix 加载，覆盖 4×16=64 行。
            ldmatrix_A<BK>(reg_a, smem.As[compute_stage], warp_id_m, lane_id, k_offset);

            // 为 B 发出四次 ldmatrix 加载，覆盖 4×8=32 列。
            ldmatrix_B<BN, BK>(reg_b, smem.Bs[compute_stage], warp_id_n, lane_id, k_offset);

            // MMA 主体：在 M、N 方向各循环 4 次，执行 m16n8k16 矩阵乘加。
            mma_compute<T>(sum, reg_a, reg_b);
        }

        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;
    }

    write_c_via_smem<BM, BN>(c, by, bx, n, sum, warp_id_m, warp_id_n, lane_id, tid, smem.Cs);
}

// ------------------- TMA 读取与 MMA 计算 -------------------
// 一个线程块计算 C 的 128×128 数据块。
template <const int BM = 128, const int BN = 128, const int BK = 64, const int STAGES = 3, typename T>
__global__ void hgemm_tma_r_k_stages_kernel(
    __grid_constant__ const CUtensorMap tma_a, __grid_constant__ const CUtensorMap tma_b, T *c, int m, int n, int k) {
    // 重排线程块到输出数据块的映射（网格 swizzle）。
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // 网格重排的逻辑分组宽度为 8。

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 动态共享内存。
    extern __shared__ __align__(128) uint8_t smem_buf[];
    T(*As)[BM][BK] = reinterpret_cast<T(*)[BM][BK]>(smem_buf);
    T(*Bs)[2][BK][BN / 2] = reinterpret_cast<T(*)[2][BK][BN / 2]>(smem_buf + STAGES * BM * BK * sizeof(T));
    T(*Cs)[BN] = reinterpret_cast<T(*)[BN]>(smem_buf);
    // mbar 数组位于共享内存缓冲末尾；每个阶段一个屏障，按 8 字节对齐。
    mbarrier_t *mbar = reinterpret_cast<mbarrier_t *>(smem_buf + BM * BK * sizeof(T) * STAGES * 2);

    // 仅 tid=0 初始化 MBarrier，期望到达数为 1；后续由该线程 arrive.expect_tx，TMA 完成则更新事务计数。
    if (tid == 0) {
        for (int i = 0; i < STAGES; ++i)
            mbarrier_init(&mbar[i], 1);
    }
    __syncthreads(); // 在线程块内协调屏障初始化；TMA 的异步代理可见性仍应按架构规范单独核对。

    // 在线程块内部，继续按线程束划分输出数据块。
    // 每个线程束负责 C 的 64×32 数据块。
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 每线程的累加器：M 方向 4 个片段 × N 方向 4 个片段 × 每片段 4 个 float，共 64 个。
    float sum[4][4][4] = {0.f};

    // 一个流水线阶段的总传输字节数：A 加上 B 的两个 TMA 分片。
    const uint32_t tx_bytes = (BM * BK + BK * BN) * sizeof(T);

    // 这里只递增数据块坐标；内存步长存放在 TMA 描述符中。
    int load_k_coord = 0;

    // 1. 流水线序言：预取 STAGES-1 个 K 数据块。
    for (int i = 0; i < STAGES - 1; ++i) {
        if (tid == 0) {
            // 此 mbarrier 阶段完成前需要观察到的传输字节数。
            mbarrier_expect_tx(&mbar[i], tx_bytes);

            cp_async_bulk_tensor_2d(&mbar[i], &tma_a, As[i], load_k_coord, by * BM);
            // Bs 沿 N 方向分成两个 TMA 数据块。
            cp_async_bulk_tensor_2d(&mbar[i], &tma_b, Bs[i][0], bx * BN, load_k_coord);
            cp_async_bulk_tensor_2d(&mbar[i], &tma_b, Bs[i][1], bx * BN + BN / 2, load_k_coord);
        }
        load_k_coord += BK;
    }

    int load_stage = STAGES - 1;
    int compute_stage = 0;
    int wait_phase = 0; // MBarrier 的阶段奇偶位，在 0、1 之间切换。
    int total_k_step = BK / 16;
    // 2. 主循环。
    for (int bk = (STAGES - 1) * BK; bk < k; bk += BK) {

        // 发起下一组 TMA，仍然只由 tid=0 执行。
        if (tid == 0) {
            mbarrier_expect_tx(&mbar[load_stage], tx_bytes);
            cp_async_bulk_tensor_2d(&mbar[load_stage], &tma_a, As[load_stage], load_k_coord, by * BM);
            cp_async_bulk_tensor_2d(&mbar[load_stage], &tma_b, Bs[load_stage][0], bx * BN, load_k_coord);
            cp_async_bulk_tensor_2d(&mbar[load_stage], &tma_b, Bs[load_stage][1], bx * BN + BN / 2, load_k_coord);
        }
        load_k_coord += BK;

        // 所有线程等待当前 compute_stage 的 TMA 数据就绪；try_wait 可在等待期间挂起线程。
        mbarrier_wait(&mbar[compute_stage], wait_phase);

        // 寄存器双缓冲：交替执行 ldmatrix 加载和 MMA 计算。
        uint32_t reg_a[2][4][4], reg_b[2][4][2];
        ldmatrix_A_tma<BK>(reg_a[0], As[compute_stage], warp_id_m, lane_id, 0);
        ldmatrix_B_tma<BN, BK>(reg_b[0], Bs[compute_stage], warp_id_n, lane_id, 0);
        int read_idx = 0, write_idx = 1;
#pragma unroll
        for (int k_step = 0; k_step < total_k_step; ++k_step) {
            if (k_step < total_k_step - 1) {
                int next_k_offset = (k_step + 1) * 16;
                ldmatrix_A_tma<BK>(reg_a[write_idx], As[compute_stage], warp_id_m, lane_id, next_k_offset);
                ldmatrix_B_tma<BN, BK>(reg_b[write_idx], Bs[compute_stage], warp_id_n, lane_id, next_k_offset);
            }
            mma_compute<T>(sum, reg_a[read_idx], reg_b[read_idx]);
            read_idx ^= 1;
            write_idx ^= 1;
        }

        // 整个线程块同步；没有单独的生产者线程束调用 arrive()。
        __syncthreads();

        // 循环轮换流水线索引。
        load_stage = (load_stage + 1 == STAGES) ? 0 : load_stage + 1;
        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;

        // compute_stage 绕回 0 时，下一轮使用相反的阶段奇偶位；此处默认共 3 个阶段。
        if (compute_stage == 0)
            wait_phase ^= 1;
    }
    // 3. 流水线收尾：处理剩余 STAGES-1 组数据。
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
        // 继续等待相应 TMA 阶段完成。
        mbarrier_wait(&mbar[compute_stage], wait_phase);

        // 寄存器双缓冲。
        uint32_t reg_a[2][4][4], reg_b[2][4][2];
        ldmatrix_A_tma<BK>(reg_a[0], As[compute_stage], warp_id_m, lane_id, 0);
        ldmatrix_B_tma<BN, BK>(reg_b[0], Bs[compute_stage], warp_id_n, lane_id, 0);
        int read_idx = 0, write_idx = 1;
#pragma unroll
        for (int k_step = 0; k_step < total_k_step; ++k_step) {
            if (k_step < total_k_step - 1) {
                int next_k_offset = (k_step + 1) * 16;
                ldmatrix_A_tma<BK>(reg_a[write_idx], As[compute_stage], warp_id_m, lane_id, next_k_offset);
                ldmatrix_B_tma<BN, BK>(reg_b[write_idx], Bs[compute_stage], warp_id_n, lane_id, next_k_offset);
            }
            mma_compute<T>(sum, reg_a[read_idx], reg_b[read_idx]);
            read_idx ^= 1;
            write_idx ^= 1;
        }

        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;
        if (compute_stage == 0)
            wait_phase ^= 1;
    }

    // 4. 写回 C。
    write_c_via_smem<BM, BN>(c, by, bx, n, sum, warp_id_m, warp_id_n, lane_id, tid, Cs);
}

// ------------------- TMA 读写与 MMA 计算（下方为尚未实现的空核函数） -------------------
// 一个线程块计算 C 的 128×128 数据块。
template <const int BM = 128, const int BN = 128, const int BK = 32, const int STAGES = 3, typename T>
__global__ void hgemm_tma_rw_k_stages_kernel(T *a, T *b, T *c, int m, int n, int k) {}

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

template <typename T, const int rowBytes = 128>
inline CUtensorMap
create_tensor_map(T *global_address, uint64_t fast_dim, uint64_t slow_dim, uint32_t fast_box, uint32_t slow_box) {
    CUtensorMap tmap;
    CUtensorMapDataType type =
        std::is_same_v<T, __half> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    CUtensorMapSwizzle swizzle = rowBytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;

    // TMA 约定：维度 0 对应内存中变化最快的维度。
    uint64_t globalDim[2] = {fast_dim, slow_dim};
    uint64_t globalStrides[1] = {fast_dim * sizeof(T)}; // 外层维度的步长，单位为字节。
    uint32_t boxDim[2] = {fast_box, slow_box};
    uint32_t elementStrides[2] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(&tmap,
                                          type,
                                          2, // 张量维数为 2，即二维矩阵。
                                          global_address,
                                          globalDim,
                                          globalStrides,
                                          boxDim,
                                          elementStrides,
                                          CU_TENSOR_MAP_INTERLEAVE_NONE,
                                          swizzle, // 共享内存的 swizzle 模式。
                                          CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed!");
    return tmap;
}

// ---------------- TMA 核函数的主机包装与绑定。
// 每个 TMA 入口分配三组 A/B 缓冲和三个 8 字节屏障，并申请动态共享内存上限。
// BK=64 与 BK=32 改变分块/资源用量；两者都由一个线程块负责 128×128 个输出。
// cuTensorMapEncodeTiled 有返回值检查，cudaFuncSetAttribute 的返回值仍沿用原实现未检查。
#define binding_tiled_tma_func_gen(name, BK)                                                                           \
    void name##_##BK(torch::Tensor a, torch::Tensor b, torch::Tensor c) {                                              \
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
        const int smem_size = BM * BK * 2 * 3 * 2 + 24;                                                                \
        if (a.dtype() == torch::kHalf) {                                                                               \
            CUtensorMap tma_a =                                                                                        \
                create_tensor_map<__half, BK * 2>(reinterpret_cast<__half *>(a.data_ptr()), K, M, BK, BM);             \
            CUtensorMap tma_b = create_tensor_map<__half>(reinterpret_cast<__half *>(b.data_ptr()), N, K, BN / 2, BK); \
                                                                                                                       \
            cudaFuncSetAttribute(                                                                                      \
                name##_kernel<BM, BN, BK, 3, __half>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);         \
            name##_kernel<BM, BN, BK, 3><<<blocks_per_grid, threads_per_block, smem_size, stream>>>(                   \
                tma_a, tma_b, reinterpret_cast<__half *>(c.data_ptr()), M, N, K);                                      \
        } else {                                                                                                       \
            CUtensorMap tma_a = create_tensor_map<__nv_bfloat16, BK * 2>(                                              \
                reinterpret_cast<__nv_bfloat16 *>(a.data_ptr()), K, M, BK, BM);                                        \
            CUtensorMap tma_b =                                                                                        \
                create_tensor_map<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16 *>(b.data_ptr()), N, K, BN / 2, BK);   \
            cudaFuncSetAttribute(                                                                                      \
                name##_kernel<BM, BN, BK, 3, __nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);  \
            name##_kernel<BM, BN, BK, 3><<<blocks_per_grid, threads_per_block, smem_size, stream>>>(                   \
                tma_a, tma_b, reinterpret_cast<__nv_bfloat16 *>(c.data_ptr()), M, N, K);                               \
        }                                                                                                              \
    }

binding_tiled_func_gen(hgemm_bcf_dbf_rw);
binding_tiled_func_gen(hgemm_k_stages);

binding_tiled_tma_func_gen(hgemm_tma_r_k_stages, 64);
binding_tiled_tma_func_gen(hgemm_tma_r_k_stages, 32);

extern void hgemm_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c);

// 导出供 Python 调用的绑定。

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(hgemm_cublas);
    torch_pybinding_func(hgemm_bcf_dbf_rw);
    torch_pybinding_func(hgemm_k_stages);
    torch_pybinding_func(hgemm_tma_r_k_stages_64);
    torch_pybinding_func(hgemm_tma_r_k_stages_32);
}
