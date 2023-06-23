#include <iostream>
#include <fstream>
#include <math.h>

#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

//#ifndef __CUDACC__
//#define __CUDACC__ 1
//#endif

// C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include/
#include <cooperative_groups.h>
//#include <sm_30_intrinsics.hpp>
namespace cg = cooperative_groups;

#include "macros.h"
#include "playground_host.h"
#include "common_funcs.cu"

#if 1
#define PRINTZ(fmt, ...) printf(fmt"\n", ##__VA_ARGS__)
#else
#define PRINTZ(fmt, ...)
#endif

template <class G>
__device__ __inline__ uint32_t get_peers(G key) {

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
#endif
}

// decoupled loop-back reduction
// https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf
template < uint32_t BlockSz >
__global__ void radixSortKernel(uint32_t *vals, uint32_t count)
{
    uint32_t thid = threadIdx.x, lane = thid % 32;
    uint32_t idx = thid + blockIdx.x * BlockSz,
            stride = BlockSz * gridDim.x;

    auto cta = cg::this_thread_block();

    // 2 bit digits: 0..3 * 8
    // 3 bit digits: 0..7 * 8
    // 4 bit digits: 0..15 * 8 => 16 bytes
    struct DigitsAcc {
        enum { num = 4 };
        uint32_t d[num]; // 4 words are enough to scan 4-bit digits
    } A = {};

    auto val = vals[idx];
    auto digit = (val % 16)*8; // (0..15) * 8

//    for(int i = 0; i < DigitsAcc::num; i++) {
//        if(digit < (i+1)*4*8)
//           A.d[i] = 1 << (digit - i*4*8);
//    }
    // NOT efficient since we summing up zeros: only 1 bit is set for
    // the whole 4 words chunk !!!
    if(digit < 4*8)
       A.d[0] = 1 << digit;
    else if(digit < 8*8)
       A.d[1] = 1 << (digit - 4*8);
    else if(digit < 12*8)
       A.d[2] = 1 << (digit - 8*8);
    else
       A.d[3] = 1 << (digit - 12*8);

    PRINTZ("%d: val: %d; A: %08X %08X %08X %08X", thid, val,
           A.d[3], A.d[2], A.d[1], A.d[0]);

    __shared__ DigitsAcc sh[BlockSz];
    // TODO: use shared mem in prefixSum !!
    auto res = prefixSum< BlockSz >(cta, A, [](auto& lhs, const auto& rhs) {
        for(int i = 0; i < DigitsAcc::num; i++)
            lhs.d[i] += rhs.d[i];
        }
    );

    if(thid == BlockSz-1) {
        // 8 | 3 | 10 | 5
        // 3 |10 |  5 | 0
        //10 | 5 |  0 | 0
        // multiply by 0x1010100
        //sh[0] = (res << 8) + (res << 16) + (res << 24);

        auto z = res;
        //   0A0B0C0D
        // * 01010101
        // = 0A0B0C0D
        // + 0B0C0D00
        // + 0C0D0000
        // + 0D000000 -> the last one is actually not needed but used for propagation

        for(int i = 0; i < DigitsAcc::num; i++) {
            if(i > 0)
                z.d[i] += z.d[i-1] >> 24; // propagate the last sum
            z.d[i] *= 0x01010101;
        }
        sh[0] = z;
    }
    cg::sync(cta);

//    int ofs = (((res + sh[0]) >> digit) & 0xFF) - 1;
//    sh[ofs] = val;

    cg::sync(cta);

    PRINTZ("%d: val: %d; pref: %X %X %X %X -- %X %X %X %X", thid, val,
           res.d[3], res.d[2], res.d[1], res.d[0],
           sh[0].d[3], sh[0].d[2], sh[0].d[1], sh[0].d[0]);

    //vals[idx] = res;
}

__device__ uint32_t g_blockCounter = 0;

// single-pass grid scan
template < uint32_t BlockSz >
__global__ void globalScanKernel(uint32_t *scanVals, uint32_t *scanTops, uint32_t count)
{
    const uint32_t s_SignalVal = 0xDEADBABE;
    uint32_t thid = threadIdx.x, lane = thid % 32, bidx = blockIdx.x;
    uint32_t idx = thid + bidx * BlockSz,
            stride = BlockSz * gridDim.x;

    auto reduceOp = [](auto& lhs, const auto& rhs) {
        lhs += rhs;
    };

    __shared__ bool amLast;
    auto cta = cg::this_thread_block();

    //for(; idx < count; idx += stride) {
        auto A = scanVals[idx];
        auto res = prefixSum< BlockSz >(cta, A, reduceOp);
        //PRINTZ("%d: %d", thid, res);
    //}
    // let each thread
    if(thid == BlockSz-1) { // data size must be block-aligned
        scanTops[bidx] = res;

        __threadfence();

        // the last param gives the maxima val
        auto inc = atomicInc(&g_blockCounter, gridDim.x);
        PRINTZ("%d incrementing", inc);
        amLast = (inc == gridDim.x - 1);
    }
    cg::sync(cta);
    if(amLast) {

        if(thid == 0)
        PRINTZ("%d Im last", bidx);
        // g_blockCounter equals  gridDim.x now!

        auto val = thid < gridDim.x ? scanTops[thid] : 0;
        auto resTop = prefixSum< BlockSz >(cta, val, reduceOp);
        if(thid < gridDim.x)
            scanTops[thid] = resTop;

        __threadfence();

        // increment once more to signal other blocks that data is ready
        if(thid == 0)
            g_blockCounter = s_SignalVal;

    } else {
        if(thid == 0) {
            auto& g_mutex = (volatile uint32_t&)g_blockCounter;
            while(g_mutex != s_SignalVal);
        }
        cg::sync(cta);
//        if(thid == 0) {
//            PRINTZ("block: %d; read tops: %d %d %d %d", blockIdx.x,
//                   scanTops[0], scanTops[1], scanTops[2], scanTops[3]);
//        }
    }
    if(bidx > 0) {
        auto top = scanTops[bidx-1];
        reduceOp(res, top);
    }
    //PRINTZ("%d: %d", idx, res);
    scanVals[idx] = res;
}

bool GPU_radixSort::launchKernel(size_t dataSize)
{
    constexpr uint32_t BlockSz = 128;
    uint32_t nblocks = (dataSize + BlockSz-1) / BlockSz;

    XPRINTZ("dataSize: %zu; #threads: %u; #blocks: %u",
            dataSize, BlockSz, nblocks);

//    void *storage = nullptr;
//    size_t bytesNeeded = 0;
//    cub::DeviceRadixSort::SortKeys(nullptr, bytesNeeded, m_cpuIndices, m_devIndices, indexSize);
    //XPRINTZ("CUB storage required: %zu", bytesNeeded);

    uint32_t *blockTops = nullptr;
    cudaMalloc(&blockTops, nblocks);

    uint32_t retCnt = 0;
    cudaMemcpyToSymbol(g_blockCounter, &retCnt, sizeof(uint32_t), 0,
                              cudaMemcpyHostToDevice);

    CU_BEGIN_TIMING(0)

    //cudaMemcpy(m_devIndices, m_cpuIndices.data(), indexSize * word_size, cudaMemcpyHostToDevice);
    //cub::DeviceRadixSort::SortKeys(storage, bytesNeeded, m_cpuIndices, m_devIndices, indexSize);

    globalScanKernel< BlockSz ><<< nblocks, BlockSz >>>(m_pinnedData, blockTops, dataSize);

    auto err = cudaGetLastError();
    XPRINTZ("cuda last error: %d -- %s", err, cudaGetErrorString(err));
    //radixSortKernel< BlockSz ><<< nblocks, BlockSz >>>(m_pinnedData, dataSize);

    //cudaMemcpy(m_cpuOut.data(), m_devOutBuf, dataSize * word_size, cudaMemcpyDeviceToHost);
    CU_END_TIMING(SetBits)

    cudaThreadSynchronize();

    cudaFree(blockTops);

    return true;
}
