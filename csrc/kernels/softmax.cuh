#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

template <typename DataType>
__device__ DataType softmax_device(
    cg::thread_block_tile<WARP_SIZE> const& warp,
    DataType score,
    int32_t lane_idx,
    int32_t n_experts
);