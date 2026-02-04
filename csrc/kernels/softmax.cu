#include "softmax.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename DataType>
__device__ DataType softmax(
    cg::thread_block_tile<WARP_SIZE> const& warp,
    DataType score,
    int32_t lane_idx,
    int32_t n_experts)
{
    float max = -INFINITY;
    if(lane_idx < n_experts){
        if(float(score) >= max){
            max = float(score);
        }
    }

    max = cg::reduce(warp, max, cg::greater<float>());

    float sum = 0.f;
    float local_score;
    if(lane_idx < n_experts){
        local_score = expf(score - max);
        sum += local_score;
    }

    sum = cg::reduce(warp, sum, cg::plus<float>());

    if(lane_idx < n_experts){
        score = static_cast<DataType>(local_score / sum);
    }
    
    return score;
}

template __device__ float softmax<float>(cg::thread_block_tile<32> const&, float, int32_t, int32_t);