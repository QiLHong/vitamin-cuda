#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "cublas_v2.h"

// cuBLAS 以列主序解释矩阵：交换 A/B 与 M/N，借助 C^T=B^T A^T 计算行主序结果。
// alpha=1、beta=0 表示覆盖 C；输入输出为 FP16/BF16，计算类型为 FP32。
void cublas_hgemm(void *A, void *B, void *C, size_t M, size_t N, size_t K, bool use_bf16 = false) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    static float alpha = 1.0;
    static float beta = 0.0;
    cudaDataType_t data_type = use_bf16 ? CUDA_R_16BF : CUDA_R_16F;
    cublasGemmEx(handle,
                 CUBLAS_OP_N,
                 CUBLAS_OP_N,
                 N,
                 M,
                 K,
                 &alpha,
                 B,
                 data_type,
                 N,
                 A,
                 data_type,
                 K,
                 &beta,
                 C,
                 data_type,
                 N,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
}

// 包装只检查 CUDA 与连续性；调用方仍须保证尺寸、dtype、同设备和输出容量匹配。
// 原实现将所有非 FP16 输入都当作 BF16，不是任意 dtype 的自动转换。
void hgemm_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(a.is_contiguous(), "Tensor A must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Tensor B must be contiguous");
    TORCH_CHECK(c.is_contiguous(), "Tensor C must be contiguous");

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    if (a.dtype() == torch::kHalf) {
        cublas_hgemm(reinterpret_cast<__half *>(a.data_ptr()),
                     reinterpret_cast<__half *>(b.data_ptr()),
                     reinterpret_cast<__half *>(c.data_ptr()),
                     M,
                     N,
                     K,
                     false);
    } else {
        cublas_hgemm(reinterpret_cast<__nv_bfloat16 *>(a.data_ptr()),
                     reinterpret_cast<__nv_bfloat16 *>(b.data_ptr()),
                     reinterpret_cast<__nv_bfloat16 *>(c.data_ptr()),
                     M,
                     N,
                     K,
                     true);
    }
}
