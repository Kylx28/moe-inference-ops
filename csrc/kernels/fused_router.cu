#include "fused_router.cuh"

template <typename T, int K>
__global__ void fused_moe_router_kernel(
    const T* __restrict__ hidden_states,
    const T* __restrict__ gate_weights,
    int* __restrict__ topk_indices,
    float* __restrict__ topk_weights,
    int d_model,
    int n_experts)
{
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);

    int token_idx = blockIdx.x * (blockDim.y) + threadIdx.y;
    int lane_id = warp.thread_rank();

    float local_score = -INFINITY;
    int local_expert_idx = -1;

    // Dot Product
    for(int i = lane_id; i < n_experts; i += WARP_SIZE){
        float dot = 0.0f;
        for(int j = 0; j < d_model; ++d){
            dot += (float)hidden_states[token_idx * d_model + j] * (float)gate_weights[i * d_model + j];
        }
        if(dot > local_score){
            local_score = dot;
            local_expert_idx = i;
        }
    }

    // Top-K


    // Softmax
    float max_val = softmax(warp, local_score, lane_id, n_experts);

    if(lane_id < K){
        topk_indices[token_idx * K + lane_id] = winner_idx;
        topk_weights[token_idx * K + lane_id] = normalized_score;
    }
}