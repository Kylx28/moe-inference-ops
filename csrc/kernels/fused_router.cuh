#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/std/limits>
#include <math.h>

namespace cg = cooperative_groups;

static constexpr int WARP_SIZE = 32;

template <typename T, int K>
__global__ void fused_moe_router_kernel(const T* __restrict__ hidden_states, const T* __restrict__ gate_weights,
                                        int* __restrict__ topk_indices, float* __restrict__ topk_weights,
                                        int d_model, int n_experts);