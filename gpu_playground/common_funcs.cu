
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


enum ShuffleType {
    stSync,
    stUp,
    stDown,
    stXor
};

template < ShuffleType Type, class NT >
__device__ __forceinline__  NT shflType(NT val, uint32_t idx,
                                   uint32_t allmsk = 0xffffffffu)
{
    constexpr uint32_t SZ = (sizeof(NT) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    union S {
        NT v;
        uint32_t d[SZ];
    };
    S in{val}, res;

    #pragma unroll
    for(uint32_t i = 0; i < SZ; i++) {

        if(Type == stSync)
            res.d[i] = __shfl_sync(allmsk, in.d[i], idx);
        else if(Type == stUp)
            res.d[i] = __shfl_up_sync(allmsk, in.d[i], idx);
        else if(Type == stDown)
            res.d[i] = __shfl_down_sync(allmsk, in.d[i], idx);
        else if(Type == stXor)
            res.d[i] = __shfl_xor_sync(allmsk, in.d[i], idx);
    }
    return res.v;
}

template < class NT >
__device__ __forceinline__  NT shflUpPred(NT val, uint32_t ofs, int32_t& pred,
                                   uint32_t allmsk = 0xffffffffu, uint32_t shfl_c = 31)
{
    constexpr uint32_t SZ = (sizeof(NT) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    union {
        NT v;
        uint32_t d[SZ];
    } in{val}, res;

     asm(R"({
        .reg .pred p;
        .reg .u32 res, pred;
        shfl.sync.up.b32 res|p, %2, %3, %4, %5;
        selp.u32 pred, 1, 0, p;
        mov.u32 %0, res;
        mov.u32 %1, pred;
        })" : "=r"(res.d[0]), "=r"(pred) : "r"(in.d[0]), "r"(ofs), "r"(shfl_c), "r"(allmsk));

    #pragma unroll
    for(uint32_t i = 1; i < SZ; i++) {
        res.d[i] = __shfl_up_sync(allmsk, in.d[i], ofs);
    }
    return res.v;
}

template < class NT >
__device__ __forceinline__  NT shflDownPred(NT val, uint32_t ofs, int32_t& pred,
                                   uint32_t allmsk = 0xffffffffu, uint32_t shfl_c = 31)
{
    constexpr uint32_t SZ = (sizeof(NT) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    union {
        NT v;
        uint32_t d[SZ];
    } in{val}, res;

     asm(R"({
        .reg .pred p;
        .reg .u32 res, pred;
        shfl.sync.down.b32 res|p, %2, %3, %4, %5;
        selp.u32 pred, 1, 0, p;
        mov.u32 %0, res;
        mov.u32 %1, pred;
        })" : "=r"(res.d[0]), "=r"(pred) : "r"(in.d[0]), "r"(ofs), "r"(shfl_c), "r"(allmsk));

    #pragma unroll
    for(uint32_t i = 1; i < SZ; i++) {
        res.d[i] = __shfl_down_sync(allmsk, in.d[i], ofs);
    }
    return res.v;
}

// sorts N arrays of type NT in one loop, NT must suppport '==' and '<' operations
template < int N, class NT, class KeyFunc >
__device__ __forceinline__  void bitonicWarpSort(uint32_t lane, NT (&V)[N], KeyFunc Key)
{
    #pragma unroll
    for(int j = 4; j >= 0; j--) {

        int SH = 1 << j;
        int bit = (lane & SH) == SH;
#if 1
        #pragma unroll
        for(int i = 0; i < N; i++)
        {
            auto X = shflType< stXor >(V[i], SH);
            int set = bit ^ (Key(V[i]) < Key(X));
            if(!(set | Key(V[i]) == Key(X))) {
                V[i] = X;
            }
        }
#else
        asm volatile(R"({
          .reg .u32 bit,dA;
          .reg .pred p, q, r;
          and.b32 bit, %1, %2;
          setp.eq.u32 q, bit, %2; // q = lane & N == N
            // is it possible to enforce swap when wA == xA for all cases ??
          setp.lt.xor.u32 p, %0, %3, q;  // p = wA.x < xA.x XOR q
          setp.eq.or.u32 p, %0, %3, p;   // p = wA.x == xA.x OR p
          @!p mov.u32 %0, %3; // if (p ^ q) == 0
        })"
        : "+r"(V) : "r"(lane), "r"(N), "r"(X));   // 3
#endif
    } // for
}

// compute prefix sum of data 'T' using reduce operation ReduceOp
template < uint32_t BlockSz, class T, class ReduceOp >
__device__ __forceinline__  T prefixSum(cg::thread_block cta, const T& data, ReduceOp op)
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
