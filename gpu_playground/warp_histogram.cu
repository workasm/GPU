#include <iostream>
#include <fstream>
#include <math.h>
#include <tuple>

#include <cuda_runtime_api.h>

#include <cooperative_groups.h>
//#include <sm_30_intrinsics.hpp>
//using namespace cooperative_groups;
namespace cg = cooperative_groups;

#include "macros.h"
#include "common_types.h"
#include "common_funcs.cu"

#if 0
#define PRINTZ(fmt, ...) printf(fmt"\n", ##__VA_ARGS__)
#else
#define PRINTZ(fmt, ...)
#endif

// inline assembler wrapper
// https://github.com/stasinek/stasm/blob/master/stk_stasm.h

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

// first we reduce C: i.e., remove duplicates and shrink C
// we assume that C is SORTED (with possible duplicates)!!
template < class NT >
__device__ __forceinline__  NT reduce_C(uint32_t lane, NT wC)
{
    const uint32_t allmsk = 0xffffffffu, shfl_c = 31;

    auto xC = __shfl_sync(allmsk, wC.x, lane - 1);
    bool leader = wC.x != xC;

    uint32_t OK = __ballot_sync(allmsk, leader),
        pos = 0, Num = lane + 1; // each thread searches Nth bit set in 'OK' (1-indexed)

    // the very elements of each group contain the sum
    for(int i = 1; i <= 16; i *= 2) { // for extreme case we also need i == 32

        uint32_t j = 16 / i;
        uint32_t mval = bfe(OK, pos, j); // extract j bits starting at pos[k] from OK[k]
        auto dif = Num - __popc(mval);
        if((int)dif > 0) {
            Num = dif, pos += j;
        }

        int32_t pred = 0;
        auto pC = shflDownPred(wC, i, pred);
        if(pred && wC.x == pC.x) {
            wC.y += pC.y;
        }
    }

    wC = shflType< stSync >(wC, pos); // read from pos-th thread
    // total cannot be 0 since we have at least some data in A
    uint32_t total = __popc(OK);
    if(lane >= total)
        wC = {};

    return wC;
}

template < auto Invalid, class NT >
struct WarpHistogramTest
{
    //using KeyType = std::invoke_result_t< KeyFunc, NT >;
    // Invalid is the value used to initialize A and B (to some large value Key(Invalid) is maximal of KeyType)
    // should reside in shared or constant memory to save registers
#define Val(s) s.y
    NT A = { Invalid },
       B = { Invalid };

    // returns 'true' if C can be successfully merged into [A,B] (enough space)
    // and 'false' otherwise
    template < class KeyFunc, class ReduceOp, class ConsumeOp >
    __device__ __forceinline__  bool operator()(int32_t lane, NT C, NT *shm, KeyFunc Key, ReduceOp Reduce, ConsumeOp Consume)
    {
        auto pA = shm, pB = shm + 32, pC = shm + 64;
        pA[lane] = A;
        pB[lane] = B;
        pC[lane] = C;
        int fullIdx = -1;
        __syncthreads();
        if(lane == 0) {
            for(int i = 0, j; i < 32; i++) {
                auto c = pC[i];
                for(j = 0; j < 64; j++) {
                    auto K = Key(shm[j]);
                    if(K == Key(c)) {
                        Reduce(shm[j], c);
                        break;
                    } else if(K == Invalid) {
                        shm[j] = c;
                        break;
                    }
                }
                if(j >= 64) { // ops no space left: need to dump 'A' and 'B' into mem
                    fullIdx = i;
                    break;
                }
            } // for i
        }
        __syncthreads();
        A = pA[lane];
        B = pB[lane];

        fullIdx = __shfl_sync(0xFFFFFFFF, fullIdx, 0); // read from lane 0
        if(fullIdx >= 0) {
            Consume(A);
            Consume(B);
            Key(B) = Invalid;
            int num = 32 - fullIdx;
            if(lane < num) {
                A = pC[fullIdx + lane];
            } else {
                Key(A) = Invalid;
            }
        }
        __syncthreads();

        PRINTZ("%d Aval/num: %d / %d; Bval/num: %d / %d; Cval/num: %d / %d",
               lane, Key(A), Val(A), Key(B), Val(B), Key(C), Val(C));

#undef Val
        return true;
    }
};

// populates warp histogram [A,B] adding a sorted sequence C to it
// KeyFunc - functor to extract comparable 'keys' from 'NT'
template < /*auto Invalid,*/ class NT, class KeyFunc, class ReduceOp >
struct WarpHistogram
{
    //using KeyType = std::invoke_result_t< KeyFunc, NT >;
    // Invalid is the value used to initialize A and B (to some large value Key(Invalid) is maximal of KeyType)
    // should reside in shared or constant memory to save registers
#define Val(s) s.y

    NT A, B;
    KeyFunc Key;
    ReduceOp Reduce;

    constexpr static uint32_t Invalid = 11111111;

    __device__ WarpHistogram(NT _A, NT _B, KeyFunc _key, ReduceOp _reduce) :
            A(_A), B(_B), Key(_key), Reduce(_reduce)
    {   }

// returns 'true' if C can be successfully merged into [A,B] (enough space)
// and 'false' otherwise
__device__ __forceinline__  bool operator()(int32_t lane, NT C)
{
    // for each thread we have 2 accumulators:
    // 1. ofs - mem ofs where to store (in sorted order)
    // 2. num and denom - the data which we store (this could be taken as 64-bit value???)

//    uint32_t eq = __ballot_sync(allmsk, C); // find delimiter threads
//    if(eq == allmsk) {
//        // perform special handling if all 'C's are equal
//        //PRINTZ("eq = %d", eq);
//    }

    const uint32_t allmsk = 0xffffffffu;

    // we have two sorted seqs [A;B] and C, merge them to produce 3 bitonic sequences
    auto revC = shflType< stSync >(C, 31 - lane);
    auto sA = A, sB = Key(B) < Key(revC) ? B : revC; // // min(B, revC);
    auto revB = shflType< stSync >(B, 31 - lane);
    auto sC = Key(C) > Key(revB) ? C : revB; // max(C, revB);

    PRINTZ("orig: %d; A: %d; B: %d; C: %d", lane, Key(A), Key(B), Key(C));
//    {
//        W wC = {C, 1};
//        auto xC = reduce_C(lane, wC);
    // question is: can we live with fragmented values ??

    // leader is the lowest thread with equal val
    // sorted: 1123344567 (at most one dup is possible)
    // leader: 1011010111
    //  idx:   0123456789
    //  we need to find the nth bit set in leader
    // pos[] = {0,2,3,5,7,8,9} - positions of set bits
    // we need:
    // lane 0: idx = 0
    // lane 1: idx = 2
    // lane 2: idx = 3
    // lane 3: idx = 5
    // we have it other way round:
    // lane 0: idx = 0
    // lane 2: idx = 1
    // lane 3: idx = 2
    // lane 5: idx = 3
//    }

    // we have 2 bitonic sequences now: [sA; sB] and sC
    // first bitonic sort step of [sA and sB] -> need
    A = Key(sA) < Key(sB) ? sA : sB;
    B = Key(sA) > Key(sB) ? sA : sB;
    C = sC;
    // we have 3 bitonic sequences: A, B and C
    // where it holds that elems of A <= elems of B <= elems of C
    // that is, [A, B, C] is partially sorted
    PRINTZ("bitonic: %d; sA: %d; sB: %d; sC: %d", lane, Key(A), Key(B), Key(C));

    NT V[3] = { A, B, C }; // warp sort three sequences in parallel
    bitonicWarpSort(lane, V, Key);
    A = V[0], B = V[1], C = V[2];

    PRINTZ("sorted: %d; sA: %d; sB: %d; sC: %d", lane, Key(A), Key(B), Key(C));

    uint32_t idx = lane - 1;
    auto xA = shflType< stSync >(Key(A), idx);
    auto xB = shflType< stSync >(Key(B), idx);
    auto xC = shflType< stSync >(Key(C), idx);

    bool leader[3] = { Key(A) != xA,
                       Key(B) != (lane == 0 ? xA : xB),
                       Key(C) != (lane == 0 ? xB : xC) };

    // if(leaderC != 0) => need to flash leaderC, i.e. write it to mem
    //PRINTZ("%d leader: [%d: %d; %d]", lane, leader[0], leader[1], leader[2]);

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

        int32_t idx = lane + i, pred = 0;
        // we do not need wrap around for pA: hence could use predicate
        auto pA = shflDownPred(A, i, pred);
        auto pB = shflType< stSync >(B, idx);
        auto pC = shflType< stSync >(C, idx);

        auto cmpA = pred ? pA : pB;
        auto cmpB = pred ? pB : pC;
        if(Key(A) == Key(cmpA))
            Reduce(A, cmpA);
        if(Key(B) == Key(cmpB))
            Reduce(B, cmpB);
        if(pred & (Key(C) == Key(pC)))
            Reduce(C, pC);
    }

    PRINTZ("%d posA: %d; val/num: %d / %d; posB: %d; val/num: %d / %d; posC: %d val/num: %d / %d",
           lane, pos[0], Key(A), Val(A), pos[1], Key(B), Val(B),
            pos[2], Key(C), Val(C));

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

    A = shflType< stSync >(A, pos[0]); // read from pos-th thread
    B = shflType< stSync >(B, pos[1]); // read from pos-th thread
    C = shflType< stSync >(C, pos[2]); // read from pos-th thread

    //  sorted:  11112 22333 33444
    //  leader:  10001 00100 00100
//    vA: 2,5,9,x,x
//    vB: A,C,F,x,x => shfl => F,x,x,A,C
//    => merge => 2,5,9,A,C F,x,x,x,x

    // total cannot be 0 since we have at least some data in A
    int32_t total = __popc(OK[0]);
    // cyclic rotate wB by total to merge results with 'A'
    B = shflType< stSync >(B, lane - total); // read from pos-th thread

    if(lane >= total) { // this indicates that the N-th thread did not find the N-th bit set
        A = B;
    }
    // valid data left in B is: (32 - totalA) - totalB
    int32_t numB = __popc(OK[1]), numC = __popc(OK[2]);
    total += numB;

    if(lane == 0)
        PRINTZ("numA = %d; numB = %d; numC = %d", __popc(OK[0]), numB, numC);

    if(total < 32) { // we still have some free space in A to move C into
        if(lane == 0)
            PRINTZ("taking wA <- wC branch");

        // cyclic rotate wC to merge with wB
        C = shflType< stSync >(C, lane - total); // read from pos-th thread
        if(lane >= total) {
            A = C;
        }
        B = C; // move the rest of C into B if any

        total += numC;
        if(lane >= total) // probably not necessary but good for debugging
            Key(A) = Invalid;

        // if total > 32 => some elements of C are left in B
        // if total <= 32 => can clear the whole B
        if(lane >= total - 32) {
            Key(B) = Invalid;
        }
        PRINTZ("%d Aval/num: %d / %d; Bval/num: %d / %d; Cval/num: %d / %d",
               lane, Key(A), Val(A), Key(B), Val(B), Key(C), Val(C));
        return true;

    // no space left in A to move C => move C into B
    } // else

    int32_t occupiedB = total - 32; // indicated how much is occupied in B, must be >= 0
    // numA = 25, numB = 24, total = 49
    // space left free in B is: 64 - total = 15
    // sapce occupied in B is: 32 - (64 - total) = total - 32

    if(lane == 0)
        PRINTZ("taking wA <- wB <- wC branch, total: %d, occupiedB: %d", total, occupiedB);

    C = shflType< stSync >(C, lane - occupiedB); // read from pos-th thread
    if(lane >= occupiedB) { // this indicates that the N-th thread did not find the N-th bit set
        B = C;
    }
    occupiedB += numC;

    if(lane == 0)
        PRINTZ("occupiedB = %d", occupiedB);

    if(lane >= occupiedB) {
        Key(B) = Invalid;
    }

    int32_t leftC = occupiedB - 32; // leftC == total - 64: amount of elements left free in C
    //! if(leftC > 0) => this indicated overflow !! i.e. the histogram is full, it must be uploaded
    bool ok = leftC <= 0; // OK if C can fit completely into [A, B]
    if(lane == 0 && leftC == 1 && Key(C) == Invalid) // its only for 0th thread but it's fine for us
        ok = true;

    PRINTZ("%d Aval/num: %d / %d; Bval/num: %d / %d; Cval/num: %d / %d; leftC: %d = %d",
           lane, Key(A), Val(A), Key(B), Val(B), Key(C), Val(C), leftC, ok);

//    if(Val(A) + Val(B) + Val(C) == 111111)
//        __syncthreads();
#undef Val
    return ok;
}

}; // WarpHistogram

// merges 32-word sorted sequence A with 16-word sorted sequence B (for halfwarp)
__device__ __inline__ void merge_seqs16(uint32_t lane/*, uint32_t A, Payload& dataA,
                                        uint32_t B, const Payload& dataB*/)
{

    union Payload {
        struct {
            float num, denom;
        };
        uint64_t v;
    };

    union W {
        struct {
            uint32_t x, y;
        };
        uint64_t v;
    };

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
