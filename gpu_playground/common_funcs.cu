
#include <math.h>

// C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include/
#include <cooperative_groups.h>
//#include <sm_30_intrinsics.hpp>

namespace cg = cooperative_groups;

#include "macros.h"


// compute prefix sum of data 'T' using reduce operation ReduceOp
template < uint32_t BlockSz, class T, class ReduceOp >
__device__ T prefixSum(cg::thread_block cta, const T& data, ReduceOp op)
{
    constexpr uint32_t warpSz = 32, allmsk = 0xffffffffu,
              postScanSz = BlockSz / warpSz;
    const uint32_t thid = threadIdx.x, lane = thid % warpSz;

    auto part = cg::tiled_partition<warpSz>(cta);

    union UT {
        enum { size = (sizeof(T) + 3)/4 };
        T data;
        uint32_t words[size];
    };

    __shared__ UT mem[postScanSz];
    //auto active = cg::coalesced_threads();

    UT X = { data };
    for(uint32_t delta = 1; delta < warpSz; delta *= 2) {

        UT tmp;
        tmp.data = part.shfl_up(X.data, delta);
        if(lane >= delta) {
            op(X.data, tmp.data); // call our reduce operation
        }
    }

    if(part.thread_rank() == warpSz-1) {
        // group rank ???
        mem[thid/warpSz] = X;
    }
    cg::sync(cta);

    //  part.meta_group_rank(), part.thread_rank()
    //printf("%d: data = %d; X.data = %d\n", thid, data, X.data);

    // only do this for the first warp
    if(part.meta_group_rank() == 0) {

        auto X2 = part.thread_rank() < postScanSz ? mem[thid] : UT{};
        for(uint32_t delta = 1; delta < postScanSz; delta *= 2) {

            UT tmp;
            tmp.data = part.shfl_up(X2.data, delta);

//            for(uint32_t i = 0; i < UT::size; i++) {
//                tmp.words[i] = __shfl_up_sync(allmsk, X2.words[i], delta, postScanSz);
////                        part.shfl_up(X2.words[i], delta);
//            }
            if(lane >= delta) {
                op(X2.data, tmp.data); // call our reduce operation
            }
        }
        if(part.thread_rank() < postScanSz)
            mem[thid] = X2;
    }
    cg::sync(cta);

    auto warpID = part.meta_group_rank();
    if(warpID > 0) {
        auto X2 = mem[warpID-1];
        op(X.data, X2.data);
    }
    return X.data;
}
