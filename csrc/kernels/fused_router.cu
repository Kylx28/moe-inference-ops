#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

static constexpr int WARP_SIZE = 32;

template <typename DataType>
__device__ DataType moe_softmax(cg::thread_block_tile<WARP_SIZE> const& warp, DataType score, int32_t laneIdx, int32_t NumTopExperts)
{

}