// ============================================================================
//
// Copyright (c) 2001-2008 Max-Planck-Institut Saarbruecken (Germany).
// All rights reserved.
//
// this file is not part of any library ;-)
//
// ----------------------------------------------------------------------------
//
// Library       : CUDA MP
//
// File          : 
//
// Author(s)     : Pavel Emeliyanenko <asm@mpi-sb.mpg.de>
//
// ============================================================================

#ifndef _FLAGS_H_
#define _FLAGS_H_

#define CUMP_PREFETCH_FROM_CUDA_ARRAY 0 // use 2D texturing
#define CUMP_USE_PAGELOCKED_MEM       1 // allocate page-locked mem
#define CUMP_USE_PTX_ASSEMBLY         1 // use PTX assembly instead of
                                        // intrinsics
#define CUMP_USE_ATOMICS              1 // use atomic intructions
#define CUMP_VERBOSE  1

#define CUMP_USE_32BIT_MODULI_SET 1   // whether to use 24 or 32-bit moduli

#define CUMP_BENCHMARK_CALLS 0  // benchmark separate kernels
#define CUMP_MEASURE_MEM 1      // measure time for GPU-host memory transfer

#define CUMP_DEVICE_MODULI_SZ 16382
#define CUMP_MODULI_STRIDE    4     // 4 elements per modulus

#define CUMP_COMPILE_DEBUG_KERNEL      1

#endif // _FLAGS_H_
