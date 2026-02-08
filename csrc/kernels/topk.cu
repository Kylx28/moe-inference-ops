// RadixSort Based Top-K

#include "topk.cuh"

// Taken from: https://github.com/AlpinDale/topk_tests/blob/88903e51516ac5a502501dc79b49bef73cf2542f/topk.cu#L17
/**
 * @brief Helper to calculate the order-preserving bin index for a float.
 * * Maps a float to a 16-bit unsigned integer such that the relative 
 * order is preserved, then buckets it into the histogram range.
 **/
static inline __device__ uint16_t extractBinIdx(float x)
{
  union {
    __half h;
    uint16_t u16;
  } tmp;
  tmp.h = __float2half_rn(x);
  tmp.u16 = (x < 0.f) ? (~tmp.u16 & 0xffff) : (tmp.u16 | 0x8000);
  return 511 - (tmp.u16 >> 7);
}

/**
 * Performs a high-performance histogram-based Top-K selection.
 * 1. Build a coarse-grained histogram to find a threshold bin.
 * 2. Prune elements and perform a localized Radix Sort on the threshold bin.
 * 
 * @tparam kNumThreads Number of threads per block (must be a power of 2).
 * @tparam kNumBins    Number of histogram buckets (default 512).
 * @tparam kTopK       The 'K' in Top-K.
 * @param logits      Pointer to the input data (usually log-probabilities).
 * @param out_indices Pointer to the output buffer for Top-K indices.
 * @param n_elements  Total number of elements in the input array.
 * 
 **/
template <int kNumThreads, int kNumBins, int kTopK>
__global__ void topk_histogram_kernel(
    const float* __restrict__ logits, 
    int* __restrict__ out_indices,
    int n_elements)
{
    static constexpr int num_final_items = 4096;

    // Initialize shared memory buffers
    __shared__ int shared_mem_histogram[kNumBins];
    __shared__ int shared_mem_threshold_bin[1];
    __shared__ int shared_mem_indices[kTopK];
    __shared__ int shared_mem_final_idx[1];

    using BlockScan = cub::BlockScan<int, kNumThreads>;
    using BlockSort = cub::BlockRadixSort<float, kNumThreads, num_final_items/kNumThreads, int>;
    
    // Initialize arrays for prefix sum offset
    int prefix_sum{0};
    int final_sum{0};

    // Shared memory config using union
    __shared__ union {
        typename BlockScan::TempStorage scan_storage;
        typename BlockSort::TempStorage sort_storage;
        struct {
            float logits[num_final_items];
            int indices[num_final_items];
        } final_items;
    } shared_mem_data;

    // Histogram Pass
    // Each thread handles one bin (same number of threads and bins)
    if(threadIdx.x < kNumBins){
        shared_mem_histogram[threadIdx.x] = 0;
    }

    __syncthreads();

    // Increment Histogram Per Thread
    for(int i = threadIdx.x; i < n_elements; i += kNumThreads){
        float logit = logits[i];
        uint16_t idx = extractBinIdx(logit);
        atomicAdd(&shared_mem_histogram[idx], 1);
    }

    __syncthreads();

    // Prefix Sum
    int bin{0};
    if(threadIdx.x < kNumBins){
        bin = shared_mem_histogram[threadIdx.x];
    }

    __syncthreads();

    BlockScan(shared_mem_data.scan_storage).ExclusiveSum(bin, prefix_sum, final_sum);

    if(threadIdx.x < kNumBins){
        shared_mem_histogram[threadIdx.x] = prefix_sum;
    }

    __syncthreads();

    // Threshold Bin
    if(threadIdx.x < kNumBins){
        int next_prefix;
        if(threadIdx.x == kNumBins - 1){
            next_prefix = final_sum;
        }
        else{
            next_prefix = shared_mem_histogram[threadIdx.x + 1];
        }

        if(prefix_sum < kTopK && next_prefix >= kTopK){
            shared_mem_threshold_bin[0] = threadIdx.x;
        }
    }

    __syncthreads();

    int threshold_bin_idx = shared_mem_threshold_bin[0];

    // Collection
    if(threadIdx.x == 0){
        shared_mem_final_idx[0] = 0;
    }

    __syncthreads();

    for(int i = threadIdx.x; i < n_elements; i += kNumThreads){
        float logit = logits[i];
        uint16_t idx = extractBinIdx(logit);

        if(idx < threshold_bin_idx){
            int destination_idx = atomicAdd(&shared_mem_histogram[idx], 1);
            if(destination_idx < kTopK){
                shared_mem_indices[destination_idx] = i;
            }
        }
        else if(idx == threshold_bin_idx){
            int destination_idx = atomicAdd(&shared_mem_final_idx[0], 1);
            if(destination_idx < num_final_items){
                shared_mem_data.final_items.logits[destination_idx] = logit;
                shared_mem_data.final_items.indices[destination_idx] = i;

            }
        }
    }

    __syncthreads();

    // Sort and Merge
    static constexpr int num_final_items_per_thread = num_final_items / kNumThreads;
    int threshold_count = shared_mem_final_idx[0];
    int needed = kTopK - shared_mem_histogram[kNumBins - 1];

    if(needed > 0){
        float final_logits[num_final_items_per_thread];
        int final_indices[num_final_items_per_thread];

    #pragma unroll
        for(int i = 0; i < num_final_items_per_thread; ++i){
            final_logits[i] = -FLT_MAX;
        }
    
    #pragma unroll
        for(int i = 0; i < num_final_items_per_thread; ++i){
            int source_idx = i * kNumThreads + threadIdx.x;
            if(source_idx < threshold_count){
                final_logits[i] = shared_mem_data.final_items.logits[source_idx];
                final_indices[i] = shared_mem_data.final_items.indices[source_idx];

            }
        }

        BlockSort(shared_mem_data.sort_storage).SortDescendingBlockedToStriped(final_logits, final_indices);

    #pragma unroll
        for(int i = 0; i < num_final_items_per_thread; ++i){
            int source_idx = i * kNumThreads + threadIdx.x;
            if(source_idx < threshold_count){
                shared_mem_data.final_items.logits[source_idx] = final_logits[i];
                shared_mem_data.final_items.indices[source_idx] = final_indices[i];
            }
        }

        __syncthreads();

        for(int i = threadIdx.x; i < needed; i += kNumThreads){
            shared_mem_indices[shared_mem_histogram[kNumBins - 1] + i] = shared_mem_data.final_items.indices[i];
        }
    }

    __syncthreads();

    // Copy back to shared memory
    #pragma unroll
    for(int i = 0; i < num_final_items_per_thread; i++){
        int local_idx = i * kNumThreads + threadIdx.x;
        if(local_idx < kTopK){
            out_indices[local_idx] = shared_mem_indices[local_idx];
        }
    }
}

torch::Tensor topk_histogram_forward(torch::Tensor logits, int k) {
    auto out_indices = torch::empty({k}, logits.options().dtype(torch::kInt32));
    topk_histogram_kernel<512, 512, 2048><<<1, 512>>>(
        logits.data_ptr<float>(), out_indices.data_ptr<int>(), logits.size(0));
    return out_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &topk_histogram_forward, "TopK Histogram Forward");
}