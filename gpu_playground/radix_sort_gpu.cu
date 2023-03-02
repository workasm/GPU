#include <iostream>
#include <fstream>
#include <math.h>

#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

//#ifndef __CUDACC__
#define __CUDACC__ 1
//#endif

// C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include/
#include <cooperative_groups.h>
//#include <sm_30_intrinsics.hpp>
using namespace cooperative_groups;

#include "macros.h"
#include "playground_host.h"

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

__global__ void set_bits_baseline(uint32_t N, const uint32_t *indices,
                                uint32_t *outBuf)
{
    uint32_t thid = threadIdx.x;
    uint32_t idx = thid + blockIdx.x * blockDim.x,
            stride = blockDim.x * gridDim.x;

    for(; idx < N / 4; idx += stride) {
        uint4 val = ((const uint4 *)indices)[idx];

        auto keyA = val.x / 32, bitA = val.x % 32,
             keyB = val.y / 32, bitB = val.y % 32,
             keyC = val.z / 32, bitC = val.z % 32,
             keyD = val.w / 32, bitD = val.w % 32;

        auto peerA = get_peers(keyA),
             peerB = get_peers(keyB);

        if(blockIdx.x == 0 && thid < 32) {
            PRINTZ("%d: peers: %d / 0x%x and %d / 0x%x -- %d %d", thid, keyA, peerA, keyB, peerB, keyC, keyD);
        }

        atomicOr(outBuf + keyA, 1u << bitA);
        atomicOr(outBuf + keyB, 1u << bitB);
        atomicOr(outBuf + keyC, 1u << bitC);
        atomicOr(outBuf + keyD, 1u << bitD);
    }
}

bool GPU_radixSort::launchKernel(size_t dataSize, size_t indexSize)
{

    uint32_t nthreads = 128, nblocks = indexSize / (4*nthreads);

    XPRINTZ("dataSize: %zu; indexSize: %zu; #threads: %u; #blocks: %u",
            dataSize, indexSize, nthreads, nblocks);

    void *storage = nullptr;
    size_t bytesNeeded = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, bytesNeeded, m_cpuIndices, m_devIndices, indexSize);
    XPRINTZ("CUB storage required: %zu", bytesNeeded);

    cudaMalloc(&storage, bytesNeeded);

    CU_BEGIN_TIMING(4)

    //cudaMemcpy(m_devIndices, m_cpuIndices.data(), indexSize * word_size, cudaMemcpyHostToDevice);
    cudaMemset(m_devOutBuf, 0, dataSize * word_size);

    //cub::DeviceRadixSort::SortKeys(storage, bytesNeeded, m_cpuIndices, m_devIndices, indexSize);

    set_bits_baseline<<< nblocks, nthreads >>>(indexSize, m_devIndices, m_devOutBuf);

    cudaMemcpy(m_cpuOut.data(), m_devOutBuf, dataSize * word_size, cudaMemcpyDeviceToHost);
    CU_END_TIMING(SetBits)

    cudaFree(storage);

    return true;
}
