#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/std/limits>
#include <math.h>

namespace cg = cooperative_groups;

static constexpr int WARP_SIZE = 32;

/**
 * @brief Numerically stable warp-level softmax.
 * * Implementation must be explicitly instantiated in the .cu file 
 * for the desired DataTypes (float, half, nv_bfloat16).
 */
template <typename DataType>
__device__ DataType softmax(
    cg::thread_block_tile<32> const& warp, 
    DataType score,
    int32_t lane_idx, 
    int32_t n_experts
);