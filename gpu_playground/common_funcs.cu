
#include <math.h>

// C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include/
#include <cooperative_groups.h>
//#include <sm_30_intrinsics.hpp>

namespace cg = cooperative_groups;

#include "macros.h"

__device__ __forceinline__ float divApprox(float a, float b) {
    float res;
    asm volatile(R"( {
        div.full.f32 %0, %1, %2;
    })" : "=f"(res) : "f"(a), "f"(b));
    return res;
}

__device__ __forceinline__ double divApprox(double a, double b) {
    double res;
    asm volatile(R"( {
        div.rn.f64 %0, %1, %2;
    })" : "=d"(res) : "d"(a), "d"(b));
    return res;
}

// extracts bitfield from src of length 'width' starting at startIdx
__device__ __forceinline__ uint32_t bfe(uint32_t src, uint32_t startIdx, uint32_t width)
{
    uint32_t bit;
    asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(bit) : "r"(src), "r"(startIdx), "r"(width));
    return bit;
}

// compute prefix sum of data 'T' using reduce operation ReduceOp
template < uint32_t BlockSz, class T, class ReduceOp >
__device__ T prefixSum(cg::thread_block cta, const T& data, ReduceOp op)
{
    constexpr uint32_t warpSz = 32, postScanSz = BlockSz / warpSz;
    const uint32_t thid = threadIdx.x, lane = thid % warpSz;

    auto part = cg::tiled_partition<warpSz>(cta);

    __shared__ T mem[postScanSz];
    //auto active = cg::coalesced_threads();

    auto X = data;
    for(uint32_t delta = 1; delta < warpSz; delta *= 2) {

        auto tmp = part.shfl_up(X, delta);
        if(lane >= delta) {
            op(X, tmp); // call our reduce operation
        }
    }

    if(part.thread_rank() == warpSz-1) {
        // group rank ???
        mem[thid/warpSz] = X;
    }
    cg::sync(cta);

    // only do this for the first warp
    if(part.meta_group_rank() == 0) {

        bool cc = part.thread_rank() < postScanSz;
        auto X2 = cc ? mem[thid] : T{};
        for(uint32_t delta = 1; delta < postScanSz; delta *= 2)
        {
            auto tmp = part.shfl_up(X2, delta);
            if(lane >= delta) {
                op(X2, tmp); // call our reduce operation
            }
        }
        if(cc)
            mem[thid] = X2;
    }
    cg::sync(cta);

    auto warpID = part.meta_group_rank();
    if(warpID > 0) {
        auto Xup = mem[warpID-1];
        op(X, Xup);
    }
    return X;
}
