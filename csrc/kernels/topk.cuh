#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

/**
 * @brief Performs a high-performance histogram-based Top-K selection.
 * * This kernel uses a two-pass pruning strategy:
 * 1. Build a coarse-grained histogram to find a threshold bin.
 * 2. Prune elements and perform a localized Radix Sort on the threshold bin.
 * * @tparam kNumThreads Number of threads per block (must be a power of 2).
 * @tparam kNumBins    Number of histogram buckets (default 512).
 * @tparam kTopK       The 'K' in Top-K.
 * * @param logits      Pointer to the input data (usually log-probabilities).
 * @param out_indices Pointer to the output buffer for Top-K indices.
 * @param n_elements  Total number of elements in the input array.
 */
template <int kNumThreads, int kNumBins, int kTopK>
__global__ void topk_histogram_kernel(
    const float* __restrict__ logits, 
    int* __restrict__ out_indices,
    int n_elements
);

/**
 * @brief Helper to calculate the order-preserving bin index for a float.
 * * Maps a float to a 16-bit unsigned integer such that the relative 
 * order is preserved, then buckets it into the histogram range.
 */
static inline __device__ uint16_t extractBinIdx(float x);

torch::Tensor topk_histogram_forward(torch::Tensor logits, int k);