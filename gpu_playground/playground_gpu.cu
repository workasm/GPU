#include <iostream>
#include <fstream>
#include <math.h>

#include <cuda_runtime_api.h>

//#ifndef __CUDACC__
#define __CUDACC__ 1
//#endif

// C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include/
#include <cooperative_groups.h>
//#include <sm_30_intrinsics.hpp>
using namespace cooperative_groups;

#include "macros.h"
#include "playground_host.h"

#define CUPRINTZ(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

using ReduceType = GPU_playground::ReduceType;

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

template < uint32_t BLOCK_SZ >
__global__ void warp_vote_kernel(const ReduceType *in, ReduceType *out, uint32_t dataSz) {

#if 0
    constexpr uint32_t warpSz = 32;
    __shared__ ReduceType mem[BLOCK_SZ/warpSz];

    thread_group g = this_thread_block();
    auto warp = tiled_partition(g, warpSz);

    uint32_t thid = g.thread_rank(),
            X = blockIdx.x * BLOCK_SZ + thid;

    ReduceType val = in[X];
    for(uint32_t delta = 1; delta < warpSz; delta <<= 1) {

        val = shuffle_up(val, delta);
    }

    if(warp.thread_rank() == warpSz-1) {
        // group rank ???
        mem[thid/warpSz] = val;
    }
    g.sync();

    constexpr uint32_t postScanSz = BLOCK_SZ / warpSz;
    if(thid < postScanSz) {
        auto v2 = mem[thid];
        for(uint32_t delta = 1; delta < postScanSz; delta <<= 1) {
            v2 = shuffle_up(v2, delta);
        }
        mem[thid] = v2;
    }
    g.sync();

    if(thid >= warpSz) { // for all warps except the first one
        auto z = thid/warpSz;
        val += mem[z - 1];
    }

    int localPrefix = -1;
    if(thid % 3 != 0) {
        uint32_t lanemask = (1 << (thid % 32)) - 1;
        localPrefix = __popc(__activemask() & lanemask);

        auto active = coalesced_threads();
        localPrefix = active.thread_rank();//active.size(); // __popc(__activemask())
    }
    // stream compation across a warp !!!
    // localPrefix is an increasing index only for active threads !!!

    out[X] = localPrefix;
#endif
}

bool GPU_reduce::launchKernel(size_t dataSz)
{
    // we can also write here CPU_pinned_mem_ptr
    // since for 64-bit addressing all page-locked memory is portable
    auto devIn = (ReduceType *)DEV_pinned_mem_ptr,
         devOut = devIn + dataSz;

    constexpr uint32_t nThreads = 256;
    dim3 threads(nThreads, 1);
    size_t shm_size = 16;
    dim3 grid((dataSz + nThreads - 1)/nThreads, 1);

    XPRINTZ("grid: (%d;%d), threads: (%d;%d); shm_size: %zd words",
            grid.x, grid.y, threads.x, threads.y, shm_size);

    float ms = 0;
    CU_BEGIN_TIMING(4)

    // TODO: reading the same data over and over again from pinned memory is inefficient!!!
    // try using memcopy instead..
    warp_vote_kernel<nThreads><<< grid, threads >>>(
                devIn, devOut, dataSz);

    CU_END_TIMING(ms)
    XPRINTZ("--- time elapsed: %.3f ms", ms);

    return true;
}

//struct DeviceMatrix {
//    Real *data;
//    uint32_t w, h, stride;
//    __device__ Real get(int32_t x, int32_t y) const {
//        return data[x + y * stride];
//    }
//};

// returns (i;j)-th submatrix
__device__ DeviceMatrix getBlock(const DeviceMatrix M, uint32_t x, uint32_t y)
{
    return DeviceMatrix{ M.data + (y*M.stride + x), M.w, M.h, M.stride  };
}

// gridDim.xyz : dimensions of a grid
// blockIdx.xyz : block index within a grid
// blockDim.xyz : dimensions of a block
// threadIdx.xyz : thread index within a block

// block size TILE_SZ x TILE_SZ
#if 1
template < uint32_t TILE_SZ >
CUMP_LAUNCH_BOUNDS(256, 8)
__global__ void mat_mul_kernel(const DeviceMatrix A, const DeviceMatrix B, float *devC)
{
#if 0
    uint32_t thX = threadIdx.x, thY = threadIdx.y,
            bidx = blockIdx.x, bidy = blockIdx.y;
    uint32_t col = bidx * TILE_SZ + thX,
             row = bidy * TILE_SZ + thY;

    extern __shared__ float sh[];
    auto shA = sh, shB = shA + TILE_SZ * TILE_SZ;

    bool ok = row < A.h && col < B.w;
    float sum = 0;
    for(uint32_t pos = 0; pos < A.w; pos += TILE_SZ)
    {
        auto xA = getBlock(A, pos, bidy*TILE_SZ);
        auto xB = getBlock(B, bidx*TILE_SZ, pos);

        auto dif = min(A.w - pos, TILE_SZ);

        if(thX < dif && row < A.h) // load only a part of the last block
            shA[thY*TILE_SZ + thX] = xA.get(thX, thY);

        if(thY < dif && col < B.w)
            shB[thY*TILE_SZ + thX] = xB.get(thX, thY);

        __syncthreads();
        for(uint32_t i = 0; i < dif; i++) {
            sum += shA[i + thY*TILE_SZ] * shB[thX + i*TILE_SZ];
        }
        __syncthreads();
    }
    if(ok)
        devC[row * B.stride + col] = sum;
#endif
}
#else

template < uint32_t TILE_SZ >
CUMP_LAUNCH_BOUNDS(256, 8)
__global__ void mat_mul_kernel(const DeviceMatrix A,
                               const DeviceMatrix B,
                               Real *devC) {

    // [wA x hA] * [wB x hB] = [B.w; A.h] and A.w == B.h

    //auto g = this_thread_block();

    extern __shared__ Real shared[];
    // cx: [0..wB], cy: [0..hA]
    const uint32_t thX = threadIdx.x, thY = threadIdx.y,
             bx = TILE_SZ, by = TILE_SZ,
             thid = thY * bx + thX,
             bidx_x = blockIdx.x, bidx_y = blockIdx.y;

    const uint32_t col = bidx_x * TILE_SZ + thX,
                   row = bidx_y * TILE_SZ + thY;

    auto shA = (Real *)shared, shB = shA + bx*by;

    // one thread block processes one horizontal stripe of A and one vertical stripe of B
    Real sum = 0;
    for(uint32_t pos = 0; pos < A.w; pos += TILE_SZ) {

        auto subA = getBlock(A, pos, bidx_y*TILE_SZ),
             subB = getBlock(B, bidx_x*TILE_SZ, pos);

        //Asub.data = &A.data[TILE_SZ*(A.stride * row + col)];
        //A.stride * bidx_y * TILE_SZ + i*TILE_SZ

        bool Aok = pos + thX < A.w && row < A.h,
             Bok = col < B.w && pos + thY < B.h;

        if(Aok)
            shA[thid] = subA.get(thX, thY);

        if(Bok)
            shB[thid] = subB.get(thX, thY);

        CU_SYNC

        uint32_t num = min(TILE_SZ, A.w - pos);
        for(uint32_t e = 0; e < num; e++) {

            Real valA = shA[e + thY*TILE_SZ], // shA of size TILE_SZ x TILE_SZ
                 valB = shB[thX + e*TILE_SZ];
            sum += valA * valB;
        }
        CU_SYNC
    }
    auto ofs = row * B.stride + col;
    if(ofs < A.h * B.stride)
        devC[ofs] = sum;

   // CUPRINTZ("%d %d -- %d", X, Y, Y * B.stride + X);
}
#endif

bool GPU_matrixMul::launchKernel()
{
    auto szA = m_A.size(), szB = m_B.size(),
         szC = m_C.size();

    // we can also write here CPU_pinned_mem_ptr
    // since for 64-bit addressing all page-locked memory is portable
    auto devA = (Real *)DEV_mem_ptr,
         devB = devA + szA, devC = (Real *)DEV_pinned_mem_ptr;

    constexpr uint32_t tileX = 16;

    // memory to preload 2 matrices
    dim3 threads(tileX, tileX);
    size_t nThids = threads.x * threads.y,
           shm_size = nThids * 2;

    dim3 grid((m_C.width() + tileX - 1)/tileX,
              (m_C.height() + tileX - 1)/tileX);

    XPRINTZ("grid: (%d;%d), threads: (%d;%d); shm_size: %zd words",
            grid.x, grid.y, threads.x, threads.y, shm_size);

    cudaMemcpy(devA, m_A.data(), szA * word_size, cudaMemcpyDefault);
    cudaMemcpy(devB, m_B.data(), szB * word_size, cudaMemcpyDefault);

    CU_BEGIN_TIMING(4)

    // TODO: reading the same data over and over again from pinned memory is inefficient!!!
    // try using memcopy instead..
    mat_mul_kernel< tileX ><<< grid, threads, shm_size * word_size >>>(
                DeviceMatrix{devA, (uint32_t)m_A.width(), (uint32_t)m_A.height(), (uint32_t)m_A.stepT()},
                DeviceMatrix{devB, (uint32_t)m_B.width(), (uint32_t)m_B.height(), (uint32_t)m_B.stepT()},
                devC);

    CU_END_TIMING("MatMul")

    //cudaMemcpy(R, devR, mem_size_out, cudaMemcpyDeviceToHost);
    return true;
}

