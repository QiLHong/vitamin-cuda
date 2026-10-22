/*
 * 中文导读：通过 CUB 的设备级基数排序处理 BF16 键，输入与输出使用独立缓冲。
 * 先查询临时工作区字节数，再分配 CUDA 字节张量，并在指定流中执行排序。
 * 包装验证 BF16、连续性及同形状，空输入直接返回；设备一致性和重叠关系仍需调用方保证。
 */
#include <cstdint>

#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

// 排序位范围为 [0,16)，覆盖 BF16 键的全部 16 位；数值排序细节由 CUB 的类型处理决定。
void cub_sort_bf16(__nv_bfloat16 *d_in, __nv_bfloat16 *d_out, int n, cudaStream_t stream) {
    const auto device = at::cuda::current_device();
    size_t temp_storage_bytes = 0;

    // 空工作区指针表示查询所需容量，这一步尚未提供实际排序工作区。
    cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes, d_in, d_out, n, 0, 16, stream);

    // 使用当前设备上的 UInt8 张量保存工作区，容量单位是字节。
    auto temp_storage = torch::empty({static_cast<int64_t>(temp_storage_bytes)},
                                     torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device));

    // 提供工作区后在 stream 上执行排序；本包装没有显式同步到 CPU。
    cub::DeviceRadixSort::SortKeys(temp_storage.data_ptr<uint8_t>(), temp_storage_bytes, d_in, d_out, n, 0, 16, stream);
}

void cub_sort(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "cub_sort only supports bfloat16");
    TORCH_CHECK(b.scalar_type() == torch::kBFloat16, "cub_sort only supports bfloat16");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input and output must have the same shape");

    const int n = static_cast<int>(a.numel());
    if (n == 0) {
        return;
    }

    cub_sort_bf16(reinterpret_cast<__nv_bfloat16 *>(a.data_ptr()),
                  reinterpret_cast<__nv_bfloat16 *>(b.data_ptr()),
                  n,
                  at::cuda::getCurrentCUDAStream());
}
