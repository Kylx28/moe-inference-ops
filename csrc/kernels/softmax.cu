#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;
#define WARP_SIZE 32
#define ROWS_PER_BLOCK 8

template <typename DataType>
__device__ DataType softmax_device(
    cg::thread_block_tile<WARP_SIZE> const& warp,
    DataType score,
    int32_t lane_idx,
    int32_t n_experts
) {
    float max_val = -INFINITY;
    if (lane_idx < n_experts) {
        max_val = float(score);
    }

    max_val = cg::reduce(warp, max_val, cg::greater<float>());

    float local_score = 0.f;
    float sum = 0.f;
    if (lane_idx < n_experts) {
        local_score = expf(float(score) - max_val);
        sum = local_score;
    }

    sum = cg::reduce(warp, sum, cg::plus<float>());

    if (lane_idx < n_experts) {
        return static_cast<DataType>(local_score / sum);
    }
    
    return static_cast<DataType>(0.0f);
}

__global__ void softmax_kernel(float* data, int n_rows, int n_cols) {
    int warp_idx_in_block = threadIdx.x / WARP_SIZE;
    int row = blockIdx.x * ROWS_PER_BLOCK + warp_idx_in_block;
    int lane_idx = threadIdx.x % WARP_SIZE;
    
    if (row >= n_rows) return;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);

    float* row_ptr = data + (row * n_cols);
    
    float val = (lane_idx < n_cols) ? row_ptr[lane_idx] : -1e20f;
    
    float result = softmax_device<float>(warp, val, lane_idx, n_cols);

    if (lane_idx < n_cols) {
        row_ptr[lane_idx] = result;
    }
}

torch::Tensor softmax_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    auto output = input.clone();
    int n_rows = input.size(0);
    int n_cols = input.size(1);

    int threads_per_block = WARP_SIZE * ROWS_PER_BLOCK;
    int blocks_in_grid = (n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    softmax_kernel<<<blocks_in_grid, threads_per_block>>>(
        output.data_ptr<float>(), n_rows, n_cols);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_forward, "Optimized Softmax Forward");
}