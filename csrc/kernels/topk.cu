#include "topk.cuh"

// Taken from: https://github.com/AlpinDale/topk_tests/blob/88903e51516ac5a502501dc79b49bef73cf2542f/topk.cu#L17
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

template <int kNumThreads, int kNumBins, int kTopK>
__global__ void topk_histogram_kernel(
    const float* __restrict__ logits, 
    int* __restrict__ out_indices,
    int n_elements)
{
    static constexpr int num_final_items = 2048;

    // Shared memory config using union

    __shared__ int shared_mem_histogram[kNumBins];
    __shared__ int shared_mem_threshold_bin[1];

    using BlockScan = cub::BlockScan<int, kNumThreads>;
    using BlockSort = cub::BlockRadixSort<float, kNumThreads, 
                                          num_final_items/kNumThreads, int>;

    __shared__ union {
        typename BlockScan::TempStorage scan_storage;
        typename BlockSort::TempStorage sort_storage;
        struct {
            float values[num_final_items];
            int indices[num_final_items];
        } finalists;
    } temp_storage;

    // Histogram Pass
    // Each thread handles one bin (same number of threads and bins)
    if(threadIdx.x < kNumBins){
        shared_mem_histogram[threadIdx.x] = 0
    }

    __syncthreads();

    // Increment Histogram Per Thread
    for(int i = threadIdx.x; i < n_elements; i += kNumThreads){
        float logit = logits[i];
        uint16_t idx = extractBinIdx(logit);
        atomicAdd(&shared_mem_histogram[idx], 1);
    }

    __syncthreads();

}