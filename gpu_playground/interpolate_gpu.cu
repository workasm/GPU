#include <iostream>
#include <fstream>
#include <math.h>
#include <tuple>

#include <cuda_runtime_api.h>
#include <cuda.h>

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
#include "warp_histogram.cu"

#if 1
#define PRINTZ(fmt, ...) printf(fmt"\n", ##__VA_ARGS__)
#else
#define PRINTZ(fmt, ...)
#endif

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
            pos = grid.thread_rank();

    const auto NaN = std::nanf(""),//CUDART_NAN_F,
               eps = (OutputReal)1e-4,
               innerRadSq = params.innerRadX * params.innerRadX,
               outerRadSq = params.outerRadX * params.outerRadX;

#if 0
    if(bidx == 0 && thX < 32) {

        using DataType = int2;
        __shared__ DataType sh[BlockSz*3];

        int32_t lane = thX;
        DataType A = {lane * 137 / 119, 1},/*253 / 129,*/
             B = {(lane + 32) * 137 / 119, 1}, // [A,B] has no duplicates
             C = {(lane + 7) * 137 / 119, 1};

        auto keyF = [](auto& V) -> auto& { return V.x; };
        auto reduceF = [](auto& lhs, const auto& rhs) { lhs.y += rhs.y; };
        auto consumeF = [](const auto& V) { PRINTZ("-- consume: %d/%d", V.x, V.y); };

        constexpr int32_t Invalid = 0xFFFFFFFF;
        WarpHistogramTest< Invalid, DataType > histo;

        if(lane >= 30) {
            keyF(histo.B) = Invalid;
            keyF(C) = 71;
        }
        if(lane == 31)
            keyF(C) = 72;

        auto warpSh = sh + (thX / 32)*32*3;
        histo(lane, C, warpSh, keyF, reduceF, consumeF);

        C = {(lane + 7) * 137 / 119, 1};
        C.x += 20;
        histo(lane, C, warpSh, keyF, reduceF, consumeF);

        C.x += 20;
        histo(lane, C, warpSh, keyF, reduceF, consumeF);

            // we are given: per-thread mem offset
            // num and denom (as data)

            // for each thread we have 2 accumulators:
            // 1. ofs - mem ofs where to store (in sorted order)
            // 2. num and denom - the data which we store (this could be taken as 64-bit value???)
    }
    return;
#else
    struct DataType {
        int32_t key;
        float num;
        float denom;
    };
    __shared__ DataType sh[BlockSz*3];
    auto warpSh = sh + (thX & ~31)*3;

    constexpr int32_t Invalid = 0xFFFFFFFF;
    WarpHistogramTest< Invalid, DataType > histo;

    auto consumeF = [devPix](const auto& V) {
        //PRINTZ("-- consume: %d (%.3f, %.3f)", V.key, V.num, V.denom);
        if(V.key != Invalid) {
            auto iptr = devPix + V.key;
            atomicAdd(&iptr->num, V.num);
            atomicAdd(&iptr->denom, V.denom);
        }
    };

#endif

    int m = 0;
    // NOTE: InterpPix can be stored as SoA (struct of arrays)
    for(uint32_t ofs = pos; ofs < nSamples; ofs += grid.size()) {

        auto wh = params.w * params.h;
        auto x = (OutputReal)devIn[ofs], y = (OutputReal)devIn[ofs + wh],
             sig = (OutputReal)devIn[ofs + wh*2];

        // since coordinates x and y are read from global mem, we can only guarantee
        // that x and y are sorted but not that they do not have dups

//        if(isnan(sig))
//            continue;

        int minX = (int)ceil(x - params.innerRadX),
            maxX = (int)floor(x + params.innerRadX),
            minY = (int)ceil(y - params.innerRadY),
            maxY = (int)floor(y + params.innerRadY);

        //PRINTZ("%d: minmax %d %d %d %d", thX, minX, maxX, minY, maxY);

        for(int iy = minY; iy <= maxY; iy++)
        {
            for(int ix = minX; ix <= maxX; ix++, m++)
            {
                OutputReal dx = ix - x, dy = iy - y,
                           wd = dy * dy + dx * dx,
                           denom = (OutputReal)1 / wd;

                bool ok = (uint32_t)iy < params.h && (uint32_t)ix < params.w;
                auto memofs = ok && !isnan(sig) ? iy * params.w + ix : Invalid;
                // if signal is nan: do not add it at all..
                denom = 1; // hacky
#if 1
                auto keyF = [](auto& V) -> auto& { return V.key; };
                auto reduceF = [](auto& lhs, const auto& rhs)
                { lhs.num += rhs.num, lhs.denom += rhs.denom; };

                // NOTE: you should also dump the rest of 'C' at the end of the loop...
                // ok is per-thread variable !!!

                int lane = thX % 32;
                DataType C{ memofs, sig * denom, denom };
                histo(lane, C, warpSh, keyF, reduceF, consumeF);
#else
                auto iptr = devPix + memofs;
                if(ok) {
                    atomicAdd(&iptr->num, sig * denom);
                    atomicAdd(&iptr->denom, denom);
                }
#endif
                continue;
#if 0
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
#endif
            } // for ix
        } // for iy
    } // for ofs

    consumeF(histo.A);
    consumeF(histo.B);
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

    const auto NaN = std::nanf("");

    //extern __shared__ InputReal sh[];
    for(uint32_t ofs = pos; ofs < total; ofs += grid.size()) {
        auto pix = devPix[ofs];
        auto denom = isnan(pix.denom) ? OutputReal(1) : pix.denom;
        devOut[ofs] = (pix.num == 0 && pix.denom == 0) ? NaN : pix.num / pix.denom;//divApprox(pix.num, denom);
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

    constexpr uint32_t BlockSz1 = 32, PerBlock1 = 4, shm_size = 0;
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

    //float fill = std::nanf("");
    //cuMemsetD32((CUdeviceptr)devPixMem, (int32_t&)(fill), dev_mem_size/sizeof(uint32_t));
    cudaMemset(devPixMem, 0, dev_mem_size);
    // TODO: reading the same data over and over again from pinned memory is inefficient!!!
    // try using memcopy instead..

    interpolate_stage1< BlockSz1, InputReal, OutputReal ><<< grid1, threads1 >>>
                   (p, m_nSamples, devIn, devPixMem);


    interpolate_stage2< BlockSz2, pixPerThread, OutputReal ><<< grid2, threads2 >>>
                   (p, m_nSamples, devPixMem, devOut);

    CU_CHECK_ERROR(cudaPeekAtLastError())
    CU_CHECK_ERROR(cudaDeviceSynchronize())

    //size_t outSz = p.w * p.h * p.numSignals * sizeof(OutputReal);
    //CU_CHECK_ERROR(cudaMemcpy(m_devOutCPU, devPixMem, std::min(dev_mem_size, outSz), cudaMemcpyDeviceToHost));

    CU_END_TIMING(GPUinterp)
    return true;
}
