/*
 * 中文导读：先筛 Top-K，再在这些候选上计算 softmax 和 Top-P，最后按截断分布采样。
 * batched 为每行分配一个线程块；split_k 先把词表分给多个块，再合并各块候选。
 * 128 位加载一次读取 8 个 16 位元素，只判断起点；要求 vocab_size 为 8 的倍数并满足对齐。
 * 重要原有限制：虽有 BF16 分派，两个加载核均通过 tmp.h[x] 按 FP16 解码打包内容，
 * 因而不能把现有 BF16 路径视作已正确支持；这里记录问题，不修改原实现。
 * 支持的 top_k 以 switch 为准：1、2、4、8、16、20；英文错误消息与实际集合不一致。
 * 有限有效分数、非空词表、合理 top_p、dtype 和尺寸等前提未得到完整检查。
 * 本次未进行 GPU 运行、采样分布验证或性能复测。
 */
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "../common/pack.cuh"

// 稳定插入排序：分数降序排列，仅在严格更大时插入；相等分数保留已有候选的先后次序。
template <const int TOP_K>
__device__ __forceinline__ bool
insert_sorted(float (&score)[TOP_K], int (&token_id)[TOP_K], float new_val, int new_id) {
    if (new_val <= score[TOP_K - 1])
        return false;

#pragma unroll
    for (int i = TOP_K - 1; i > 0; i--) {
        bool bigger_than_curr = (new_val > score[i]);
        bool bigger_than_prev = (new_val > score[i - 1]);

        score[i] = bigger_than_curr ? (bigger_than_prev ? score[i - 1] : new_val) : score[i];
        token_id[i] = bigger_than_curr ? (bigger_than_prev ? token_id[i - 1] : new_id) : token_id[i];
    }

    bool bigger_than_0 = (new_val > score[0]);
    score[0] = bigger_than_0 ? new_val : score[0];
    token_id[0] = bigger_than_0 ? new_id : token_id[0];
    return true;
}

template <const int TOP_K = 20, const int CHUNK_SIZE = 2048, const int BLOCK_SIZE = 256, typename T>
__global__ void sampling_topk_topp_batched_kernel(
    T *logits, int *output_ids, float top_p, int64_t seed, int64_t offset, int vocab_size) {
    const int batch_id = blockIdx.x;
    const int tid = threadIdx.x * 8;

    T *row_ptr = logits + batch_id * vocab_size;
    // 第 1 步：各线程维护局部有序的分数/词元编号数组。
    float score[TOP_K];
    int token_id[TOP_K];
#pragma unroll
    for (int i = 0; i < TOP_K; i++) {
        score[i] = -FLT_MAX;
        token_id[i] = -1;
    }
    for (int idx = tid; idx < vocab_size; idx += CHUNK_SIZE) {
        pack128 tmp;
        tmp.f4 = FLOAT4(row_ptr[idx]);
        for (int x = 0; x < 8; x++) {
            float val = static_cast<float>(tmp.h[x]);
            int v_idx = idx + x;
            if (val > score[TOP_K - 1]) {
                insert_sorted<TOP_K>(score, token_id, val, v_idx);
            }
        }
    }

    // 第 2 步：在线程束内规约合并各线程的局部候选数组。
    int lane_id = threadIdx.x % 32;
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {

#pragma unroll
        for (int j = 0; j < TOP_K; ++j) {
            float other_val = __shfl_down_sync(0xffffffff, score[j], delta);
            int other_id = __shfl_down_sync(0xffffffff, token_id[j], delta);

            if (lane_id < delta) {
                insert_sorted<TOP_K>(score, token_id, other_val, other_id);
            }
        }
    }
    // 第 3 步：完成线程块内的最终合并，并执行采样。
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    __shared__ float smem_warp_score[num_warps][TOP_K];
    __shared__ int smem_warp_id[num_warps][TOP_K];

    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            smem_warp_score[warp_id][i] = score[i];
            smem_warp_id[warp_id][i] = token_id[i];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {

        // 线程 0 合并各线程束的 Top-K 候选。
        for (int w = 1; w < num_warps; w++) {
#pragma unroll
            for (int j = 0; j < TOP_K; ++j) {
                float other_val = smem_warp_score[w][j];
                int other_id = smem_warp_id[w][j];
                if (!insert_sorted<TOP_K>(score, token_id, other_val, other_id)) {
                    break;
                }
            }
        }

        // 仅对筛出的 Top-K 分数计算 softmax 的未归一化概率。
        float max_val = score[0];
        float sum_prob = 0.0f;
        float probs[TOP_K];
#pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            probs[i] = expf(score[i] - max_val);
            sum_prob += probs[i];
        }

        // 在 Top-K 内按降序累计归一化概率，保留累计概率首次达到 top_p 的前缀。
        float cumsum = 0.0f;
        float trunc_sum = 0.0f;
        int last_idx = TOP_K - 1;

        for (int i = 0; i < TOP_K; i++) {
            float p = probs[i] / sum_prob;
            cumsum += p;
            if (cumsum >= top_p) {
                last_idx = i;
                trunc_sum = cumsum;
                break;
            }
        }
        if (trunc_sum == 0.0f)
            trunc_sum = cumsum;

        // 初始化 cuRAND 的 Philox 随机数状态。
        curandStatePhilox4_32_10_t state;
        curand_init(seed, blockIdx.x, offset, &state);
        float u = curand_uniform(&state); // 生成取值范围为 (0, 1] 的浮点随机数。
        float random_val = 1.f - u;

        // 根据截断后的概率累计分布选择词元。
        float r = random_val * trunc_sum;
        float cdf = 0.0f;
        int final_id = token_id[last_idx];

        for (int i = 0; i <= last_idx; i++) {
            cdf += probs[i] / sum_prob;
            if (cdf >= r) {
                final_id = token_id[i];
                break;
            }
        }

        // 写回最终采样得到的词元编号。
        output_ids[blockIdx.x] = final_id;
    }
}

// split-k 第一阶段：每个词表分片独立选出局部 Top-K。
template <const int TOP_K = 20, const int CHUNK_SIZE = 2048, const int BLOCK_SIZE = 256, typename T>
__global__ void sampling_topk_topp_split_k_pass1_kernel(T *logits, int vocab_size, float *ws_score, int *ws_id) {
    const int split_id = blockIdx.x;
    const int batch_id = blockIdx.y;
    const int num_splits = gridDim.x;

    const int vocab_per_split = ((vocab_size + num_splits * 8 - 1) / (num_splits * 8)) * 8;
    const int vocab_start = split_id * vocab_per_split;
    const int vocab_end = min(vocab_start + vocab_per_split, vocab_size);

    T *row_ptr = logits + batch_id * vocab_size;
    int tid = threadIdx.x * 8;
    // 第 1 步：各线程维护局部有序的分数/词元编号数组。
    float score[TOP_K];
    int token_id[TOP_K];
#pragma unroll
    for (int i = 0; i < TOP_K; i++) {
        score[i] = -FLT_MAX;
        token_id[i] = -1;
    }
    for (int idx = vocab_start + tid; idx < vocab_end; idx += CHUNK_SIZE) {
        pack128 tmp;
        tmp.f4 = FLOAT4(row_ptr[idx]);
        for (int x = 0; x < 8; x++) {
            float val = static_cast<float>(tmp.h[x]);
            int v_idx = idx + x;
            if (val > score[TOP_K - 1]) {
                insert_sorted<TOP_K>(score, token_id, val, v_idx);
            }
        }
    }
    // 第 2 步：在线程束内规约合并各线程的局部候选数组。
    int lane_id = threadIdx.x % 32;
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {
#pragma unroll
        for (int j = 0; j < TOP_K; ++j) {
            float other_val = __shfl_down_sync(0xffffffff, score[j], delta);
            int other_id = __shfl_down_sync(0xffffffff, token_id[j], delta);
            if (lane_id < delta) {
                insert_sorted<TOP_K>(score, token_id, other_val, other_id);
            }
        }
    }
    // 第 3 步：在线程块内合并候选，并写入全局内存工作区。
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    __shared__ float smem_warp_score[num_warps][TOP_K];
    __shared__ int smem_warp_id[num_warps][TOP_K];
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            smem_warp_score[warp_id][i] = score[i];
            smem_warp_id[warp_id][i] = token_id[i];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // 线程 0 合并各线程束的 Top-K 候选。
        for (int w = 1; w < num_warps; w++) {
#pragma unroll
            for (int j = 0; j < TOP_K; ++j) {
                float other_val = smem_warp_score[w][j];
                int other_id = smem_warp_id[w][j];
                if (!insert_sorted<TOP_K>(score, token_id, other_val, other_id)) {
                    break;
                }
            }
        }
        int ws_offset = (batch_id * num_splits + split_id) * TOP_K;
        for (int i = 0; i < TOP_K; i++) {
            ws_score[ws_offset + i] = score[i];
            ws_id[ws_offset + i] = token_id[i];
        }
    }
}

// split-k 第二阶段：合并所有分片的 Top-K 候选，再执行采样。
template <const int TOP_K = 20, const int BLOCK_SIZE = 256>
__global__ void sampling_topk_topp_split_k_pass2_kernel(
    int *output_ids, float top_p, int64_t seed, int64_t offset, int num_splits, float *ws_score, int *ws_id) {
    const int batch_id = blockIdx.x;
    const int total = num_splits * TOP_K;
    const int ws_base = batch_id * total;

    float score[TOP_K];
    int token_id[TOP_K];
#pragma unroll
    for (int i = 0; i < TOP_K; i++) {
        score[i] = -FLT_MAX;
        token_id[i] = -1;
    }

    for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
        insert_sorted<TOP_K>(score, token_id, ws_score[ws_base + idx], ws_id[ws_base + idx]);
    }

    // 在线程束内规约合并候选。
    int lane_id = threadIdx.x % 32;
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {
#pragma unroll
        for (int j = 0; j < TOP_K; ++j) {
            float other_val = __shfl_down_sync(0xffffffff, score[j], delta);
            int other_id = __shfl_down_sync(0xffffffff, token_id[j], delta);
            if (lane_id < delta) {
                insert_sorted<TOP_K>(score, token_id, other_val, other_id);
            }
        }
    }

    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    __shared__ float smem_warp_score[num_warps][TOP_K];
    __shared__ int smem_warp_id[num_warps][TOP_K];
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            smem_warp_score[warp_id][i] = score[i];
            smem_warp_id[warp_id][i] = token_id[i];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {

        // 线程 0 合并各线程束的 Top-K 候选。
        for (int w = 1; w < num_warps; w++) {
#pragma unroll
            for (int j = 0; j < TOP_K; ++j) {
                float other_val = smem_warp_score[w][j];
                int other_id = smem_warp_id[w][j];
                if (!insert_sorted<TOP_K>(score, token_id, other_val, other_id)) {
                    break;
                }
            }
        }

        // 仅对筛出的 Top-K 分数计算 softmax 的未归一化概率。
        float max_val = score[0];
        float sum_prob = 0.0f;
        float probs[TOP_K];
#pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            probs[i] = expf(score[i] - max_val);
            sum_prob += probs[i];
        }

        // 在 Top-K 内按降序累计归一化概率，保留累计概率首次达到 top_p 的前缀。
        float cumsum = 0.0f;
        float trunc_sum = 0.0f;
        int last_idx = TOP_K - 1;
        for (int i = 0; i < TOP_K; i++) {
            float p = probs[i] / sum_prob;
            cumsum += p;
            if (cumsum >= top_p) {
                last_idx = i;
                trunc_sum = cumsum;
                break;
            }
        }
        if (trunc_sum == 0.0f)
            trunc_sum = cumsum;

        // 初始化 cuRAND 的 Philox 随机数状态。
        curandStatePhilox4_32_10_t state;
        curand_init(seed, batch_id, offset, &state);
        float u = curand_uniform(&state);
        float random_val = 1.f - u;

        // 根据截断后的概率累计分布选择词元。
        float r = random_val * trunc_sum;
        float cdf = 0.0f;
        int final_id = token_id[last_idx];
        for (int i = 0; i <= last_idx; i++) {
            cdf += probs[i] / sum_prob;
            if (cdf >= r) {
                final_id = token_id[i];
                break;
            }
        }

        // 写回最终采样得到的词元编号。
        output_ids[batch_id] = final_id;
    }
}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

// 这里只对 FP16/BF16 分派，其他 dtype 没有报错分支，可能返回尚未写入的 res。
// BF16 的位模式解码限制见文件开头；调用方不能仅凭存在分派分支认定结果正确。
#define DISPATCH_TOPK_KERNEL(kernel_name, K_VAL)                                                                       \
    if (logits.dtype() == torch::kHalf) {                                                                              \
        kernel_name<K_VAL, chunk_size>                                                                                 \
            <<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<__half *>(logits.data_ptr()),         \
                                                                reinterpret_cast<int *>(res.data_ptr()),               \
                                                                top_p,                                                 \
                                                                seed,                                                  \
                                                                offset,                                                \
                                                                vocab_size);                                           \
    } else if (logits.dtype() == torch::kBFloat16) {                                                                   \
        kernel_name<K_VAL, chunk_size>                                                                                 \
            <<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<__nv_bfloat16 *>(logits.data_ptr()),  \
                                                                reinterpret_cast<int *>(res.data_ptr()),               \
                                                                top_p,                                                 \
                                                                seed,                                                  \
                                                                offset,                                                \
                                                                vocab_size);                                           \
    }

#define binding_tiled_func_gen(name)                                                                                   \
    torch::Tensor name(torch::Tensor logits, int top_k, float top_p, int64_t seed, int64_t offset) {                   \
        CHECK_T(logits);                                                                                               \
        const int bs = logits.size(0);                                                                                 \
        const int vocab_size = logits.size(1);                                                                         \
        constexpr int threads_per_block = 256;                                                                         \
        constexpr int chunk_size = threads_per_block * 8;                                                              \
        const dim3 blocks_per_grid(bs);                                                                                \
        auto options = torch::TensorOptions().dtype(torch::kInt32).device(logits.device());                            \
        auto res = torch::empty({bs}, options);                                                                        \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        switch (top_k) {                                                                                               \
            case 1: DISPATCH_TOPK_KERNEL(name##_kernel, 1); break;                                                     \
            case 2: DISPATCH_TOPK_KERNEL(name##_kernel, 2); break;                                                     \
            case 4: DISPATCH_TOPK_KERNEL(name##_kernel, 4); break;                                                     \
            case 8: DISPATCH_TOPK_KERNEL(name##_kernel, 8); break;                                                     \
            case 16: DISPATCH_TOPK_KERNEL(name##_kernel, 16); break;                                                   \
            case 20: DISPATCH_TOPK_KERNEL(name##_kernel, 20); break;                                                   \
            default: TORCH_CHECK(false, "Unsupported top_k! Only powers of 2 up to 32 are supported."); break;         \
        }                                                                                                              \
        return res;                                                                                                    \
    }

#define DISPATCH_TOPK_SPLITK_KERNEL(kernel_name, K_VAL)                                                                \
    if (logits.dtype() == torch::kHalf) {                                                                              \
        kernel_name<K_VAL, chunk_size>                                                                                 \
            <<<blocks_p1, threads_per_block, 0, stream>>>(reinterpret_cast<__half *>(logits.data_ptr()),               \
                                                          vocab_size,                                                  \
                                                          ws_score.data_ptr<float>(),                                  \
                                                          ws_id.data_ptr<int>());                                      \
        sampling_topk_topp_split_k_pass2_kernel<K_VAL>                                                                 \
            <<<blocks_p2, threads_per_block, 0, stream>>>(reinterpret_cast<int *>(res.data_ptr()),                     \
                                                          top_p,                                                       \
                                                          seed,                                                        \
                                                          offset,                                                      \
                                                          num_splits,                                                  \
                                                          ws_score.data_ptr<float>(),                                  \
                                                          ws_id.data_ptr<int>());                                      \
    } else if (logits.dtype() == torch::kBFloat16) {                                                                   \
        kernel_name<K_VAL, chunk_size>                                                                                 \
            <<<blocks_p1, threads_per_block, 0, stream>>>(reinterpret_cast<__nv_bfloat16 *>(logits.data_ptr()),        \
                                                          vocab_size,                                                  \
                                                          ws_score.data_ptr<float>(),                                  \
                                                          ws_id.data_ptr<int>());                                      \
        sampling_topk_topp_split_k_pass2_kernel<K_VAL>                                                                 \
            <<<blocks_p2, threads_per_block, 0, stream>>>(reinterpret_cast<int *>(res.data_ptr()),                     \
                                                          top_p,                                                       \
                                                          seed,                                                        \
                                                          offset,                                                      \
                                                          num_splits,                                                  \
                                                          ws_score.data_ptr<float>(),                                  \
                                                          ws_id.data_ptr<int>());                                      \
    }

// num_splits 固定为 26；它改变并行划分，不等同于运行时查询 SM 数量。
// 工作区存放各词表分片的 top_k 个分数与编号，同一流上的第二阶段继续合并。
torch::Tensor sampling_topk_topp_split_k(torch::Tensor logits, int top_k, float top_p, int64_t seed, int64_t offset) {
    CHECK_T(logits);
    const int bs = logits.size(0);
    const int vocab_size = logits.size(1);
    constexpr int threads_per_block = 256;
    constexpr int chunk_size = threads_per_block * 8;
    constexpr int num_splits = 26;
    const dim3 blocks_p1(num_splits, bs);
    const int blocks_p2 = bs;
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(logits.device());
    auto res = torch::empty({bs}, options);
    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(logits.device());
    auto ws_score = torch::empty({bs * num_splits * top_k}, float_opts);
    auto ws_id = torch::empty({bs * num_splits * top_k}, options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (top_k) {
        case 1: DISPATCH_TOPK_SPLITK_KERNEL(sampling_topk_topp_split_k_pass1_kernel, 1); break;
        case 2: DISPATCH_TOPK_SPLITK_KERNEL(sampling_topk_topp_split_k_pass1_kernel, 2); break;
        case 4: DISPATCH_TOPK_SPLITK_KERNEL(sampling_topk_topp_split_k_pass1_kernel, 4); break;
        case 8: DISPATCH_TOPK_SPLITK_KERNEL(sampling_topk_topp_split_k_pass1_kernel, 8); break;
        case 16: DISPATCH_TOPK_SPLITK_KERNEL(sampling_topk_topp_split_k_pass1_kernel, 16); break;
        case 20: DISPATCH_TOPK_SPLITK_KERNEL(sampling_topk_topp_split_k_pass1_kernel, 20); break;
        default: TORCH_CHECK(false, "Unsupported top_k! Only powers of 2 up to 32 are supported."); break;
    }
    return res;
}

binding_tiled_func_gen(sampling_topk_topp_batched);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(sampling_topk_topp_batched);
    torch_pybinding_func(sampling_topk_topp_split_k);
}
