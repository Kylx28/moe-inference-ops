#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

template <int kNumThreads, int kNumBins, int kTopK>
__global__ void topk_histogram_kernel(
    const float* __restrict__ logits, 
    int* __restrict__ out_indices,
    int n_elements
);

static inline __device__ uint16_t extractBinIdx(float x);

torch::Tensor topk_histogram_forward(torch::Tensor logits, int k);