#include <iostream>
#include <fstream>
#include <math.h>

#include <cuda_runtime_api.h>

//#ifndef __CUDACC__
//#define __CUDACC__ 1
//#endif

#include <cooperative_groups.h>
//#include <sm_30_intrinsics.hpp>
//using namespace cooperative_groups;
namespace cg = cooperative_groups;

#include "macros.h"
#include "playground_host.h"
#include "common_types.h"

__device__ float divApprox(float a, float b) {
    float res;
    asm volatile(R"( {
        div.full.f32 %0, %1, %2;
    })" : "=f"(res) : "f"(a), "f"(b));
    return res;
}

__device__ double divApprox(double a, double b) {
    double res;
    asm volatile(R"( {
        div.rn.f64 %0, %1, %2;
    })" : "=d"(res) : "d"(a), "d"(b));
    return res;
}

template <class G>
__device__ __inline__ uint32_t get_peers(G key) {

    // warps.size() is the number of active threads in a warp participating in this call
//    if(bidx == 0 && thX < 32) {
//        printf("num_threads: %d; thread_rank: %d\n",
//               warps.size(),warps.thread_rank());
//    }
    uint32_t peers = 0;
    bool is_peer;

#if 0
    do {
        auto warp = cg::coalesced_threads();
        // fetch key of first unclaimed lane and compare with this key
        is_peer = (key == warp.shfl(key, 0));

        // determine which lanes had a match
        peers = warp.ballot(is_peer);
        // remove lanes with matching keys from the pool
        //unclaimed ^= peers;

        // quit if we had a match
    } while (!is_peer);
#else
    // in the beginning, all lanes are available
    uint32_t mask = 0xffffffffu, unclaimed = __activemask();
    do {
        // fetch key of first unclaimed lane and compare with this key
        is_peer = (key == __shfl_sync(mask, key, __ffs(unclaimed) - 1));

        // determine which lanes had a match
        peers = __ballot_sync(mask, is_peer);

        // remove lanes with matching keys from the pool
        unclaimed ^= peers;

        // quit if we had a match
    } while (!is_peer);
#endif
    return peers;
}

// merges 32-word sorted sequence A with 16-word sorted sequence C (for halfwarp)
__device__ __inline__ void merge_seqs16(uint32_t thX)
{
    uint32_t A = thX * 7 / 57,
             C = thX * 11 / 57;
    if(thX >= 16)
        C = 0x1000000;  // this values shall be ignored..

    printf("original thX: %d; sA: %d; sC: %d\n", thX, A, C);

    // addrA, valA, addrC, valC
    uint32_t allmsk = 0xffffffffu;

    uint32_t revC = __shfl_sync(allmsk, C, 31 - thX);
    uint32_t sA = min(A, revC);

    uint32_t revA = __shfl_sync(allmsk, A, 31 - thX);
    uint32_t sC = max(C, revA);

    // bitonic sort sA and sC sequences
    for(int N = 16; N > 0; N /= 2) {

        auto xA = __shfl_xor_sync(allmsk, sA, N);
        auto xC = __shfl_xor_sync(allmsk, sC, N);

        if(thX & N) {
            sA = max(sA, xA);
            sC = max(sC, xC);
        } else {
            sA = min(sA, xA);
            sC = min(sC, xC);
        }
    }
    printf("sort thX: %d; sA: %d; sC: %d\n", thX, sA, sC);

    union W {
        struct {
            uint32_t x, y;
        };
        uint64_t v;
    };

    W wA = { sA, 1 },
      wC = { sC, 1 };

    // 0123 45 6789A BCD - thid
    // 0000 11 22222 333 - val
    // 0001 01 00001 001 - ballot
    // thread[i] takes the index of the ith non-zero bit
    // thread[0] = 3
    // thread[1] = 5
    // thread[2] = 10, etc.
    // __popc() - total # of set bits

    // the very elements of each group contain the sum
    for(int i = 1; i <= 16; i *= 2) { // for extreme case we also need i == 32
        W pA, pC;
        uint32_t idx = thX + i;
        pA.v = __shfl_sync(allmsk, wA.v, idx);
        pC.v = __shfl_sync(allmsk, wC.v, idx % 32);
        if(idx < 32) {
            if(wA.x == pA.x)
                wA.y += pA.y;
            if(wC.x == pC.x)
                wC.y += pC.y;
        } else {
            if(wA.x == pC.x)
                wA.y += pC.y;
        }
    }
    // finally merge wA and wC
    printf("thX: %d; valA: %d; numA: %d; valC: %d; numC: %d;\n",
           thX, wA.x, wA.y, wC.x, wC.y);
}

#if 0
__device__ uint32_t shuffle_up(uint32_t val, uint32_t delta, const uint32_t mask = 0xFFFFFFFF) {

    uint32_t input = val, shfl_c = 0; // shfl_c = (width - warpSize)
    asm volatile(
      "{"
      "  .reg .u32 r0;"
      "  .reg .pred p;"
      "  shfl.sync.up.b32 r0|p, %1, %2, %3, %4;"
      "  @p add.u32 r0, r0, %1;"
      "  mov.u32 %0, r0;"
      "}"
      : "=r"(val) : "r"(input), "r"(delta), "r"(shfl_c), "r"(mask));

    //        auto newv = __shfl_up_sync(mask, val, delta);
    //        if(warp.thread_rank() >= delta)
    //            val += newv;
    return val;
}
#endif

__device__ __inline__ void sorted_seq_histogram()
{
    uint32_t tid = threadIdx.x, lane = tid % 32;
    uint32_t val = (lane + 117)* 23 / 97; // sorted sequence of values to be reduced

    printf("%d: val = %d\n", lane, val);
    uint32_t num = 1;

    uint32_t allmsk = 0xffffffffu, shfl_c = 31;
    for(int i = 1; i <= 16; i *= 2) {

#if 1
        uint32_t xval = __shfl_down_sync(allmsk, val, i),
                 xnum = __shfl_down_sync(allmsk, num, i);
        if(lane + i < 32) {
            if(val == xval)
                num += xnum;
        }
#else  // this is a (hopefully) optimized version of the code above
        asm(R"({
          .reg .u32 r0,r1;
          .reg .pred p;
          shfl.sync.down.b32 r0|p, %1, %2, %3, %4;
          shfl.sync.down.b32 r1|p, %0, %2, %3, %4;
          @p setp.eq.s32 p, %1, r0;
          @p add.u32 r1, r1, %0;
          @p mov.u32 %0, r1;
        })"
        : "+r"(num) : "r"(val), "r"(i), "r"(shfl_c), "r"(allmsk));
#endif
    }
    // shfl.sync wraps around: so thread 0 gets the value of thread 31
    bool leader = val != __shfl_sync(allmsk, val, lane - 1);
    auto OK = __ballot_sync(allmsk, leader); // find delimiter threads
    auto total = __popc(OK); // the total number of unique numbers found

    auto lanelt = (1 << lane) - 1;
    auto idx = __popc(OK & lanelt);

    printf("%d: val = %d; num = %d; total: %d; idx = %d; leader: %d\n", lane, val, num, total, idx, leader);

    // OK has ones for threads at the end of the sequence
    __shared__ uint32_t sh[64];
    if(leader) {
        sh[idx] = val;
        sh[idx + 32] = num;
    }
    __syncthreads();

    if(lane < total) {
        val = sh[lane], num = sh[lane + 32];
    } else {
        val = 0xDEADBABE, num = 0;
    }
    printf("%d: final val = %d; num = %d\n", lane, val, num);
}

/*

now we have [A; B] and [C; D] where D = +infty

AB compare with revD, revC
sA = min(A, revD) = A
sB = min(B, revC)

CD compare with revB, revA
sC = max(C, revB)
sD = max(D, revA) = D

so, we have [sA; sB] and sC
 */

__device__ __inline__ void merge_seqs32(uint32_t thX)
{
    uint32_t A = thX * 7 / 19,
             B = (thX + 32) * 7 / 19,
             C = thX * 11 / 13;

    // we have two sorted seqs [A;B] and C

    uint32_t allmsk = 0xffffffffu;
    uint32_t revC = __shfl_sync(allmsk, C, 31 - thX);

    uint32_t sA = A, sB = min(B, revC);

    uint32_t revB = __shfl_sync(allmsk, B, 31 - thX);
    uint32_t sC = max(C, revB);

    printf("thX: %d; A: %d; B: %d; C: %d\n", thX, A, B, C);

    // we have 2 bitonic sequences now: [sA; sB] and sC
    // first bitonic sort step of [sA and sB] -> need
    A = min(sA, sB),
    B = max(sA, sB);
    C = sC;
    // now we have a sorted array [A, B, C] where the components are bitonic sequences
    printf("thX: %d; sA: %d; sB: %d; sC: %d\n", thX, A, B, C);

    uint2 V = make_uint2(A,B);
    // bitonic sort sA, sB and C sequences..
    for(int N = 16; N > 0; N /= 2) {

        auto xC = __shfl_xor_sync(allmsk, C, N);
        auto V8 = __shfl_xor_sync(allmsk, (uint64_t&)V, N);
        uint2 VV= (uint2&)V8;

        if(thX & N) {
            V.x = max(V.x, VV.x);
            V.y = max(V.y, VV.y);
            C = max(C, xC);
        } else {
            V.x = min(V.x, VV.x);
            V.y = min(V.y, VV.y);
            C = min(C, xC);
        }
        //printf("sort thX: %d; sA: %d; xA: %d; N = %d\n", thX, sA, xA, N);
    }
    printf("sorted thX: %d; sA: %d; sB: %d; sC: %d\n", thX, V.x, V.y, C);
#if 0
    uint32_t input = val, shfl_c = 0; // shfl_c = (width - warpSize)
    asm volatile(
      "{"
      "  .reg .u32 r0;"
      "  .reg .pred p;"
      "  shfl.sync.up.b32 r0|p, %1, %2, %3, %4;"
      "  @p add.u32 r0, r0, %1;"
      "  mov.u32 %0, r0;"
      "}"
      : "=r"(val) : "r"(input), "r"(delta), "r"(shfl_c), "r"(mask));
#endif

    auto Z = make_uint2(V.x, 1);

    auto up = __shfl_sync(allmsk, V.x, (thX + 1)%32); // so that the last thread gets 0
    auto OK = __ballot_sync(allmsk, up != V.x); // find delimiter threads
    // OK has ones for threads at the end of the sequence
    auto lanelt = (1 << thX) - 1;
    //auto before = (up != V.x ? __popc(OK & zz) : -777);
    auto before = __popc(OK & lanelt);

    //int reduce = __shfl_sync(allmsk, which one??)

//    11112223333344
//    00010010000101
//    0123456789ABCD

//    this is a ballot bitpattern
//    get elements of 3rd, 6th, 11th, 13th
//    if(val != next) {
        // last element
        // mask = (1 << lane) - 1;
        // before = __popc(ballot & mask); - this is a number of bits set below me
           //             so we need transfer to thread with ID = before
//    }
    //thread 1/3/7

//    total = popc(mask); total number of bits set


    for(int i = 1; i <= 16; i *= 2) {
        auto up = __shfl_down_sync(allmsk, (uint64_t&)Z, i);
        auto U = (uint2&)up;
        if(thX + i < 32 && U.x == Z.x) {
            Z.y += U.y;
        }
    }

    printf("thX: %d; val: %d; num: %d; leader: 0x%X; before: %d\n", thX, Z.x, Z.y, OK, before);

    if(V.x+V.y+C == 111111)
        __syncthreads();
}


template <class G>
__device__ __inline__ void reduce_peers(G key, uint32_t thX)
{

//    1. reduce all peers to get 2 registers:
//    key; val

//    if key == 0xFFFFFFFF -> meaning that this one is free

//    problem: merge 'key; val' with the next one 'key2;val2'
//    where key and key2 should generally overlap well
    auto lane = thX % 32;
    key = thX % 5 + 1;

//        idx & (1 << lane) is always true: the calling thread's bit is always set in idx

//        upper_peer = the one after 'lane'
//        lane = 4
//        mask: 11110000: ignore all lanes lower
//        threads: 0, 3, 4, 8
//        peer = 0100011001
//        // mask all uppper bits including myself
//        my_pos = __popc(peer & ((1 << lane)-1))

//        ((1<<lane)-1)
//        lane=3: mask must be: 000000111
//        lane=0: 1-1=0
//        lane=1: (1<<1)-1 = 000000001

    uint32_t val = 1;
    auto peers = get_peers(key);
    auto first = __ffs(peers) - 1; // only leading thread keeps the result
    auto mypos = __popc(peers & ((1 << lane)-1));
    auto mask = ~((1 << (lane + 1)) - 1);

    printf("lane: %d; key: %X; peers: 0x%X; mypos: %d\n",
           lane, key, peers, mypos);

    peers &= mask;

    uint32_t allmsk = 0xffffffffu;
    while(__any_sync(allmsk, peers))
    {
        auto next = __ffs(peers);
        auto t = __shfl_sync(allmsk, val, next-1);
        if(next)
            val += t;
        bool done = mypos & 1;
        auto which = __ballot_sync(allmsk, done); // every second thread is done since we are doing tree-like reduction
        peers &= ~which; // remove those which are done
        mypos >>= 1;
    }
    if(lane != first) {// if lane is not first in its group of peers, we can zero it out
        val = 0, key = 0;
    }

    printf("lane: %d; key: %X; peers: 0x%X; mypos: %d; val: %d -- %d\n",
           lane, key, peers, mypos, val, first);

#if 1
    auto key2 = thX % 7 + 1;

    // bitmask of threads having non-empty keys
    auto active = __ballot_sync(allmsk, key != 0); // lane == first
    auto which = __ffs(active)-1;
    bool is_peer = (key2 == __shfl_sync(allmsk, key, which));
    peers = __ballot_sync(allmsk, is_peer);
//    peers shows if any threads have key2 == key of 'which' thread
    uint32_t unclaimed = allmsk;
    unclaimed ^= peers;

    printf("%d: key: %d; key2: %d; which: %d; peer: %X\n",
           lane, key, key2, which, peers);

    active &= ~(1 << which);
    which = __ffs(active)-1;
    if(!is_peer) {
        // take second key...
    }


    printf("%d: key: %d; key2: %d; which: %d; unclaimed: %X\n",
           lane, key, key2, which, unclaimed);

    return;
#endif

    // normally you would have (key1, val1) and (key2, val2) for each thread
    // then comes new one (key, val) and we need to add them to already existing elements

}

// devIn: input data of nSamples where each sample has 'params.numSignals + 2' signals (or channels)
// devOut: output bitmap of size p.w * p.h with params.numSignals channels
// devPix: scratch buffer for interpolation: p.w * p.h * p.numSignals * sizeof(DataInterp::Pix);
//CUMP_LAUNCH_BOUNDS(256, 8)
template < uint32_t BlockSz, class InputReal, class OutputReal >
__global__ void interpolate_stage1(InterpParams< OutputReal > params, size_t nSamples,
                                   const InputReal *devIn, InterpPix< OutputReal > *devPix)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    // pos = grid.thread_rank() ??
    uint32_t thX = threadIdx.x, bidx = blockIdx.x,
            pos = grid.thread_rank(), stride = params.numSignals + 2;

    extern __shared__ InputReal sh[];

    const auto NaN = nanf(""),
               eps = (OutputReal)1e-4,
               innerRadSq = params.innerRadX * params.innerRadX,
               outerRadSq = params.outerRadX * params.outerRadX;

    if(bidx == 0 && thX < 32) {
        sorted_seq_histogram();
        //reduce_peers(0, thX);
    }
    return;

    int m = 0;
    // NOTE: InterpPix can be stored as SoA (struct of arrays)
    for(uint32_t ofs = pos; ofs < nSamples; ofs += grid.size()) {

        auto sh = ofs*stride;
        auto x = (OutputReal)devIn[sh], y = (OutputReal)devIn[sh + 1],
             sig = (OutputReal)devIn[sh + 2];

        if(isnan(sig))
            continue;

        int minX = max(0, (int)ceil(x - params.innerRadX)),
            maxX = min((int)params.w - 1, (int)floor(x + params.innerRadX)),
            minY = max(0, (int)ceil(y - params.innerRadY)),
            maxY = min((int)params.h - 1, (int)floor(y + params.innerRadY));

        for(int iy = minY; iy <= maxY; iy++)
        {
            for(int ix = minX; ix <= maxX; ix++, m++)
            {
                auto memofs = iy * params.w + ix;
                if(bidx == 0 && m == 0 && thX < 32) {
                 }

                auto iptr = devPix + memofs;
                const OutputReal dx = ix - x, dy = iy - y,
                           wd = dy * dy + dx * dx,
                           denom = (OutputReal)1 / wd;
                atomicAdd(&iptr->num, sig);
                atomicAdd(&iptr->denom, 1);
                continue;

                if(wd < eps) {
                    atomicExch(&iptr->denom, NaN); // this sets the exact point
                    atomicExch(&iptr->num, sig);

                } else if(wd <= innerRadSq) {
                    auto old = atomicAdd(&iptr->denom, denom); // this sets the exact point
                    if(!isnan(old)) { // if denom is NaN => the point was set exactly
                        atomicAdd(&iptr->num, sig * denom);
                    }
                } else if(wd <= outerRadSq) {
                    // but if the inner radius has already been set => do nothing
                    // deal with outer radius...
//                    auto old = atomicAdd(&iptr->denom, denom); // this sets the exact point
//                    if(!isnan(old)) { // if denom is NaN => the point was set exactly
//                        atomicAdd(&iptr->num, sig * denom);
//                    }
                }
            }
        } // for iy
    } // for ofs
}

// grid size = # of pixels
// output interpolation
template < uint32_t BlockSz, uint32_t PixPerThread, class OutputReal >
__global__ void interpolate_stage2(InterpParams< OutputReal > p, size_t nSamples,
                                   const InterpPix< OutputReal > *devPix,
                                   OutputReal *devOut)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    const uint32_t thX = threadIdx.x, bidx = blockIdx.x,
              pos = grid.thread_rank(), total = p.w * p.h;

    //extern __shared__ InputReal sh[];
    for(uint32_t ofs = pos; ofs < total; ofs += grid.size()) {
        auto pix = devPix[ofs];
        auto denom = isnan(pix.denom) ? OutputReal(1) : pix.denom;
        devOut[ofs] = pix.num / denom;//divApprox(pix.num, denom);
    }
}


bool GPU_interpolator::launchKernel(const InterpParams< OutputReal >& p)
{
     // we can also write here CPU_pinned_mem_ptr
    // since for 64-bit addressing all page-locked memory is portable

    // input and output buffers are pinned memory
    // pinned_mem_size = in.size() * sizeof(InputReal) + outImg.total() * p.numSignals * sizeof(OutputReal);
    // internal device buffer for collecting samples..
    //dev_mem_size = p.w * p.h * p.numSignals * sizeof(DataInterp::Pix);
    auto devIn = (InputReal *)DEV_pinned_mem_ptr;
    auto devOut = (OutputReal *)(devIn + m_nSamples * (p.numSignals + 2));

    using Pix = InterpPix< OutputReal >;
    auto devPixMem = (Pix *)DEV_mem_ptr;

    constexpr uint32_t BlockSz1 = 128, PerBlock1 = 4, shm_size = 0;
    dim3 threads1(BlockSz1, 1);
    dim3 grid1(m_nSamples / BlockSz1*PerBlock1);

    XPRINTZ("grid1: (%d;%d), threads1: (%d;%d); shm_size1: %zd words",
               grid1.x, grid1.y, threads1.x, threads1.y, shm_size);

    constexpr uint32_t pixPerThread = 4, // # of pixels processed by each thread
                       BlockSz2 = 128, pixPerBlock = pixPerThread * BlockSz2;

    dim3 grid2((p.w * p.h + pixPerBlock - 1) / pixPerBlock, 1),
         threads2(BlockSz2, 1);

    XPRINTZ("grid2: (%d;%d), threads2: (%d;%d); shm_size2: %zd words",
            grid2.x, grid2.y, threads2.x, threads2.y, shm_size);

    CU_BEGIN_TIMING(0)

    cudaMemset(devPixMem, 0, dev_mem_size);
    // TODO: reading the same data over and over again from pinned memory is inefficient!!!
    // try using memcopy instead..

    interpolate_stage1< BlockSz1, InputReal, OutputReal ><<< grid1, threads1, 1 * word_size >>>
                   (p, m_nSamples, devIn, devPixMem);


    interpolate_stage2< BlockSz2, pixPerThread, OutputReal ><<< grid2, threads2, 1 * word_size >>>
                   (p, m_nSamples, devPixMem, devOut);

    CU_CHECK_ERROR(cudaPeekAtLastError())
    CU_CHECK_ERROR(cudaDeviceSynchronize())

//    size_t outSz = p.w * p.h * p.numSignals * sizeof(OutputReal);
//    CU_CHECK_ERROR(cudaMemcpy(m_devOutCPU, devPixMem, std::min(dev_mem_size, outSz), cudaMemcpyDeviceToHost));

    CU_END_TIMING("GPU interp")
    return true;
}

