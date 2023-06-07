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
#include "common_funcs.cu"

#if 1
#define PRINTZ(fmt, ...) printf(fmt"\n", ##__VA_ARGS__)
#else
#define PRINTZ(fmt, ...)
#endif

// inline assembler wrapper
// https://github.com/stasinek/stasm/blob/master/stk_stasm.h

__device__ __inline__ void sorted_seq_histogram()
{
    uint32_t tid = threadIdx.x, lane = tid % 32;
    uint32_t val = (lane + 117)* 23 / 97; // sorted sequence of values to be reduced

    PRINTZ("%d: val = %d", lane, val);
    uint32_t num = 1;

    const uint32_t allmsk = 0xffffffffu, shfl_c = 31;

    // shfl.sync wraps around: so thread 0 gets the value of thread 31
    bool leader = val != __shfl_sync(allmsk, val, lane - 1);
    auto OK = __ballot_sync(allmsk, leader); // find delimiter threads
    uint32_t pos = 0, N = lane+1; // each thread searches Nth bit set in 'OK' (1-indexed)

    for(int i = 1; i <= 16; i *= 2) {

        uint32_t j = 16 / i;
        //uint32_t mval = bfe(OK, pos, j); // extract j bits starting at pos from OK
        uint32_t mval = (OK >> pos) & ((1 << j) - 1);
        auto dif = N - __popc(mval);
        if((int)dif > 0) {
            N = dif, pos += j;
        }

#if 0
        uint32_t xval = __shfl_down_sync(allmsk, val, i),
                 xnum = __shfl_down_sync(allmsk, num, i);
        if(lane + i < 32) {
            if(val == xval)
                num += xnum;
        }
#else  // this is a (hopefully) optimized version of the code above
        asm(R"({
          .reg .u32 r0,r1;
          .reg .pred p, q;
          shfl.sync.down.b32 r0|p, %1, %2, %3, %4;
          shfl.sync.down.b32 r1|p, %0, %2, %3, %4;
          setp.eq.and.s32 q, %1, r0, p;
          @q add.u32 r1, r1, %0;
          @q mov.u32 %0, r1;
        })"
        : "+r"(num) : "r"(val), "r"(i), "r"(shfl_c), "r"(allmsk));
#endif
    }
    num = __shfl_sync(allmsk, num, pos); // read from pos-th thread
    val = __shfl_sync(allmsk, val, pos); // read from pos-th thread

    auto total = __popc(OK); // the total number of unique numbers found
    if(lane >= total) {
        num = 0xDEADBABE;
    }
    PRINTZ("%d: final val = %d; num = %d", lane, val, num);
    if(num + val == 1233123)
        __syncthreads();
}

union Payload {
    struct {
        float num, denom;
    };
    uint64_t v;
};

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
    // we are given: per-thread mem offset
    // num and denom (as data)

    // for each thread we have 2 accumulators:
    // 1. ofs - mem ofs where to store (in sorted order)
    // 2. num and denom - the data which we store (this could be taken as 64-bit value???)

    const uint32_t lane = thX % 32;
    uint32_t A = lane * 33 / 129,
             B = (lane + 32) * 33 / 129, // [A,B] has no duplicates
             C = (lane + 7) * 37 / 119;

    // NOTE that A and B are already compressed: i.e.,
    // [A,B] is a sorted sequence without duplicates
    // C also has no duplicates ??

    // we have two sorted seqs [A;B] and C
    const uint32_t allmsk = 0xffffffffu, shfl_c = 31;
    uint32_t revC = __shfl_sync(allmsk, C, 31 - lane);
    uint32_t sA = A, sB = min(B, revC);
    uint32_t revB = __shfl_sync(allmsk, B, 31 - lane);
    uint32_t sC = max(C, revB);

    PRINTZ("orig: %d; A: %d; B: %d; C: %d", lane, A, B, C);

    // we have 2 bitonic sequences now: [sA; sB] and sC
    // first bitonic sort step of [sA and sB] -> need
    A = min(sA, sB),
    B = max(sA, sB);
    C = sC;
    // we have 3 bitonic sequences: A, B and C
    // where it holds that elems of A <= elems of B <= elems of C
    // that is, [A, B, C] is partially sorted
    PRINTZ("bitonic: %d; sA: %d; sB: %d; sC: %d", lane, A, B, C);

    union W {
        struct {
            uint32_t x, y;
        };
        uint64_t v;
    };

    W V = { A, B };
    // bitonic sort sA, sB and C sequences..
    for(int j = 4; j >= 0; j--) {
        int N = 1 << j;
        W xV;
        auto xC = __shfl_xor_sync(allmsk, C, N);
        xV.v = __shfl_xor_sync(allmsk, V.v, N);

        if(lane & N) {
            V.x = max(V.x, xV.x);
            V.y = max(V.y, xV.y);
            C = max(C, xC);
        } else {
            V.x = min(V.x, xV.x);
            V.y = min(V.y, xV.y);
            C = min(C, xC);
        }
    }
    PRINTZ("sorted: %d; sA: %d; sB: %d; sC: %d", lane, V.x, V.y, C);

    // leader is the lowest thread with equal val
    //  sorted:  11112223333344
    //  leader:  10001001000010
    //  idx:     0123456789ABCD

    W wA = { V.x, 1 };
    W wB = { V.y, 1 };
    W wC = { C, 1 };

    uint32_t idx = lane - 1;
    auto xA = __shfl_sync(allmsk, wA.x, idx);
    auto xB = __shfl_sync(allmsk, wB.x, idx);
    auto xC = __shfl_sync(allmsk, wC.x, idx);
    bool leader[3];
    leader[0] = wA.x != xA;
    leader[1] = wB.x != (lane == 0 ? xA : xB);
    leader[2] = wC.x != (lane == 0 ? xB : xC);

    // if(leaderC != 0) => need to flash leaderC, i.e. write it to mem
    PRINTZ("%d leader: [%d: %d; %d]", lane, leader[0], leader[1], leader[2]);

    uint32_t OK[3], pos[3], Num[3];
    for(int k = 0; k < 3; k++) {
        OK[k] = __ballot_sync(allmsk, leader[k]); // find delimiter threads
        pos[k] = 0, Num[k] = lane + 1; // each thread searches Nth bit set in 'OK' (1-indexed)
    }

    // the very elements of each group contain the sum
    for(int i = 1; i <= 16; i *= 2) { // for extreme case we also need i == 32

        uint32_t j = 16 / i;
        for(int k = 0; k < 3; k++) {
            uint32_t mval = bfe(OK[k], pos[k], j); // extract j bits starting at pos[k] from OK[k]
            auto dif = Num[k] - __popc(mval);
            if((int)dif > 0) {
                Num[k] = dif, pos[k] += j;
            }
        }

        W pA, pB, pC;
        uint32_t idx = lane + i;
        // we do not need wrap around for pA: hence could use predicate
        pA.v = __shfl_down_sync(allmsk, wA.v, i);
        pB.v = __shfl_sync(allmsk, wB.v, idx);
        pC.v = __shfl_sync(allmsk, wC.v, idx);
        if(idx < 32) {
            if(wA.x == pA.x)
                wA.y += pA.y;
            if(wB.x == pB.x)
                wB.y += pB.y;
            if(wC.x == pC.x)
                wC.y += pC.y;
        } else { // this is a case for wrap around
            if(wA.x == pB.x)
                wA.y += pB.y;
            if(wB.x == pC.x)
                wB.y += pC.y;
        }
    }
    //        A    B    C
    // val:   1122 2334 4445
    // final: 1234 5xxx xxxx
    // count: 2324 1
    // that is, leaders must be shifted left: B->A, C->B, otherwise we won't have enough space

    // pos[0] - gives thread locations to read from
    // but do we really need to move everything down ?? i.e. compress ??
    // data:   1111222333344555555
    // leader: 1000100100010100000
    // at the end, 'leader' threads contain data, remaining threads contain garbage..
    // they will be tightly packed at the end:

    wA.v = __shfl_sync(allmsk, wA.v, pos[0]); // read from pos-th thread
    wB.v = __shfl_sync(allmsk, wB.v, pos[1]); // read from pos-th thread
    wC.v = __shfl_sync(allmsk, wC.v, pos[2]); // read from pos-th thread

    //  sorted:  11112 22333 33444
    //  leader:  10001 00100 00100
//    vA = shfl(vA, pos[0]) - consolidate vA's
//    if(lane >= total)
//        vA = vB;

//    vA: 2,5,9,x,x
//    vB: A,C,F,x,x => shfl => F,x,x,A,C
//    => merge => 2,5,9,A,C F,x,x,x,x

    auto total = __popc(OK[0]);
    // cyclic rotate wB by total to merge results with 'A'
     wB.v = __shfl_sync(allmsk, wB.v, lane - total); // read from pos-th thread

    if(lane >= total) { // this indicates that the N-th thread did not find the N-th bit set
        wA = wB;
        // get data from wB.v
    }
//    total = __popc(OK[1]);
//    if(lane >= total) { //this no longer works if OK does not change
//        wB.y = 0;
//    }
//    total = __popc(OK[2]);
//    if(lane >= total) { //this no longer works if OK does not change
//        wC.y = 0;
//    }

    int tid = total, n = total+222;

    PRINTZ("%d posA: %d; val/num: %d / %d; posB: %d; val/num: %d / %d; posC: %d val/num: %d / %d",
           lane, pos[0], wA.x, wA.y, pos[1], wB.x, wB.y,
            pos[2], wC.x, wC.y);

    //PRINTZ("%d: final A = [%d; %d]; B = [%d; %d]; C = [%d; %d]", lane,
      //     wA.x, wA.y, wB.x, wB.y, wC.x, wC.y);

    if(wA.y + wB.y + wC.y == 111111)
        __syncthreads();
}


// merges 32-word sorted sequence A with 16-word sorted sequence B (for halfwarp)
__device__ __inline__ void merge_seqs16(uint32_t lane/*, uint32_t A, Payload& dataA,
                                        uint32_t B, const Payload& dataB*/)
{
    constexpr uint32_t bogus = 0x1000000; // special bogus value
    uint32_t A = 5 + lane * 27 / 57;             // base address A (sorted)
    Payload dataA, dataB;
    dataA.num = -(float)lane - 12;   // payload for A
    dataA.denom = 1.0f / dataA.num;

    uint32_t B = lane * 47 / 111;            // address B (to be merged with A)
    dataB.num = (float)lane + 17;
    dataB.denom = 1.0f / dataB.num;
//    if(lane >= 16)
//        C = bogus;  // this values shall be ignored..

    // indexing into payload data: to save on shuffles
    uint32_t idxA = lane, idxB = lane + 32;

    // 33334455556667
    // 10001010001001 - ballot
    // 0123456789abcd
//    lt = (1 << lane)-1;
//    idx = __fss(ballot & lt);
//    for each leader thread, idx gives the location where it should be written

    PRINTZ("%d: original sA: %d; sB: %d", lane, A, B);

    const uint32_t allmsk = 0xffffffffu, shfl_c = 31, idx = 31 - lane;

    uint32_t revB = __shfl_sync(allmsk, B, idx);
    uint32_t sA = A, sB = B;
    if(revB < A) { // min(revB, A)
        sA = revB;
        idxA = 31 + 32 - lane; // revIdxB = shfl(idxB, 31 - lane)
    }

    uint32_t revA = __shfl_sync(allmsk, A, idx);
    if(revA > B) { // max(B, revA);
        sB = revA;
        idxB = idx; // revIdxA = 31 - lane
    }

    union W {
        struct {
            uint32_t x, y;
        };
        uint64_t v;
    };

    // so the question: do we really need to sort all this ???

    W wA = { sA, idxA },
      wB = { sB, idxB };

    // bitonic sort sA and sC sequences
    for(int j = 4; j >= 0; j--) {

//        PRINTZ("%d / %d: half-sorted sA: %d / %d; sB: %d / %d", j, lane, wA.x, wA.y, wB.x, wB.y);

        W xA, xB;
        uint32_t N = 1 << j;
        xA.v = __shfl_xor_sync(allmsk, wA.v, N);
        xB.v = __shfl_xor_sync(allmsk, wB.v, N);
#if 0
        uint32_t bit = bfe(lane, j, 1); // checks j-th bit
        uint32_t dA = wA.x < xA.x, dB = wB.x < xB.x;
        if(bit == dA && wA.x != xA.x) // do not do anything on move
            wA = xA;
        if(bit == dB && wB.x != xB.x)
            wB = xB;
#else
        asm volatile(R"({
          .reg .u32 bit,dA;
          .reg .pred p, q, r;
          and.b32 bit, %4, %5;
          setp.eq.u32 q, bit, %5; // q = lane & N == N
            // is it possible to enforce swap when wA == xA for all cases ??
          setp.lt.xor.u32 p, %0, %6, q;  // p = wA.x < xA.x XOR q
          setp.eq.or.u32 p, %0, %6, p;   // p = wA.x == xA.x OR p
          @!p mov.u32 %0, %6; // if (p ^ q) == 0
          @!p mov.u32 %1, %7;
          setp.lt.xor.u32 p, %2, %8, q;  // p = wB.x < xB. XOR q
          setp.eq.or.u32 p, %2, %8, p;   // p = wB.x == xB.x OR p
          @!p mov.u32 %2, %8; // if (p ^ q) == 0
          @!p mov.u32 %3, %9;
        })"
        : "+r"(wA.x), "+r"(wA.y), "+r"(wB.x), "+r"(wB.y) : // 0, 1, 2, 3
            "r"(lane), "r"(N),                             // 4, 5
            "r"(xA.x), "r"(xA.y), "r"(xB.x), "r"(xB.y));   // 6, 7, 8, 9
#endif

    }
    // idx: 0 1 32 37 3 4 41 42 44 6 7
#if 0
    {
        PRINTZ("%d: sorted sA: %d / %d; sB: %d / %d; payload: %f / %f",
               lane, wA.x, wA.y, wB.x, wB.y, dataA.num, dataB.num);

        Payload xA, xAc, xB, xBc;
        xA.v = __shfl_sync(allmsk, dataA.v, wA.y);
        xAc.v = __shfl_sync(allmsk, dataB.v, wA.y - 32);
        xB.v = __shfl_sync(allmsk, dataB.v, wB.y - 32);
        xBc.v = __shfl_sync(allmsk, dataA.v, wB.y);
        dataA.v = (wA.y < 32 ? xA.v : xAc.v);
        dataB.v = (wB.y >= 32 ? xB.v : xBc.v);
        PRINTZ("%d: sorted payload: %f / %f",
               lane, dataA.num, dataB.num);
    }
#endif
//    if(wA.x+ wA.y+ wB.x+ wB.y == 991919)
//        __syncthreads();
    //return;

    // NOTE: here we should really reduce data !!!
    // i.e. take indices and load data using wA/B.y
    wA.y = 1; // drop the indices for now for testing
    wB.y = 1;

    // 0123 45 6789A BCD - thid
    // 0000 11 22222 333 - val
    // 1000 10 10000 100 - ballot
    // thread[i] takes the index of the ith non-zero bit

    uint32_t OK[2], pos[2], Num[2];
    {
        uint32_t idx = lane - 1;
        // NOTE: these two sync commands can be merged with the first step of
        // reduction algorithm
        auto xA = __shfl_sync(allmsk, wA.x, idx);
        auto xB = __shfl_sync(allmsk, wB.x, idx);
        // if(leaderC != 0) => need to flash leaderC, i.e. write it to mem
        bool leader[2];
        leader[0] = wA.x != xA;
        leader[1] = wB.x != (lane == 0 ? xA : xB);

        for(int k = 0; k < 2; k++) {
            OK[k] = __ballot_sync(allmsk, leader[k]); // find delimiter threads
            pos[k] = 0, Num[k] = lane + 1; // each thread searches Nth bit set in 'OK' (1-indexed)
        }
    }

    // the very elements of each group contain the sum
    for(int i = 1; i <= 16; i *= 2) { // for extreme case we also need i == 32

        uint32_t j = 16 / i;
        for(int k = 0; k < 2; k++) {
            uint32_t mval = bfe(OK[k], pos[k], j); // extract j bits starting at pos[k] from OK[k]
            auto dif = Num[k] - __popc(mval);
            if((int)dif > 0) {
                Num[k] = dif, pos[k] += j;
            }
        }

        // FEDC BA98 7654 3210
        // 0010 1101 0011 1000
        // 6th bit set = 12
        uint32_t idx = lane + i;
#if 1
        asm(R"({
          .reg .u32 pAx,pAy,pBx,pBy,selx,sely;
          .reg .pred p, q;
          shfl.sync.down.b32 pAx|p, %2, %4, %6, %7;  // p = lane + i < 32
          shfl.sync.down.b32 pAy|p, %0, %4, %6, %7;
          shfl.sync.idx.b32 pBx, %3, %5, %6, %7;
          shfl.sync.idx.b32 pBy, %1, %5, %6, %7;
          selp.b32 selx, pAx, pBx, p;
          selp.b32 sely, pAy, pBy, p;
          setp.eq.u32 q, %2, selx;      // q = selx == wA.x
          @q add.u32 sely, %0, sely;    // if(q) sely += wA.y
          @q mov.u32 %0, sely;
          setp.eq.and.u32 q, %3, pBx, p; // q = pBx == wB.x AND p
          @q add.u32 pBy, %1, pBy;       // if(q) pBy += += wB.y
          @q mov.u32 %1, pBy;
        })"
        : "+r"(wA.y), "+r"(wB.y) : "r"(wA.x), "r"(wB.x), // 0, 1, 2, 3
           "r"(i), "r"(idx), "r"(shfl_c), "r"(allmsk)); // 4, 5, 6, 7

#else
        W pA, pB;
        // we do not need wrap around for pA: hence could use predicate
        pA.v = __shfl_down_sync(allmsk, wA.v, i);
        pB.v = __shfl_sync(allmsk, wB.v, idx);
        if(idx < 32) {
            if(wA.x == pA.x)
                wA.y += pA.y;
            if(wC.x == pC.x)
                wC.y += pC.y;
        } else {
            if(wA.x == pC.x)
                wA.y += pC.y;
        }
#endif
    }
    // finally merge wA and wC
    PRINTZ("%d; final valA: %d; numA: %d; valB: %d; numB: %d;",
           lane, wA.x, wA.y, wB.x, wB.y);

    wA.v = __shfl_sync(allmsk, wA.v, pos[0]); // read from pos-th thread
    wB.v = __shfl_sync(allmsk, wB.v, pos[1]); // read from pos-th thread

    auto total = __popc(OK[0]);
    if(lane >= total) { // this indicates that the N-th thread did not find the N-th bit set
        wA.y = 0;
    }
    total = __popc(OK[1]);
    if(lane >= total) { //this no longer works if OK does not change
        wB.y = 0;
    }

    PRINTZ("%d posA: %d; val/num: %d / %d; posB: %d; val/num: %d / %d",
           lane, pos[0], wA.x, wA.y, pos[1], wB.x, wB.y);

    if(wA.y + wB.y == 111111)
        __syncthreads();
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

    //extern __shared__ InputReal sh[];

    const auto NaN = nanf(""),
               eps = (OutputReal)1e-4,
               innerRadSq = params.innerRadX * params.innerRadX,
               outerRadSq = params.outerRadX * params.outerRadX;

    if(bidx == 0 && thX < 32) {
        merge_seqs32(thX);
    }
    return;

    int m = 0;
    // NOTE: InterpPix can be stored as SoA (struct of arrays)
    for(uint32_t ofs = pos; ofs < nSamples; ofs += grid.size()) {

        auto sh = ofs*stride;
        // here we read from global mem with stride 3 ... grrr
        // better use vector loads..
        auto x = (OutputReal)devIn[sh], y = (OutputReal)devIn[sh + 1],
             sig = (OutputReal)devIn[sh + 2];

        // since coordinates x and y are read from global mem, we can only guarantee
        // that x and y are sorted but not that they do not have dups

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
    dim3 grid1(m_nSamples / (BlockSz1*PerBlock1));

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

    CU_END_TIMING(GPUinterp)
    return true;
}

