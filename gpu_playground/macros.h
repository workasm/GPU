
#ifndef _MACROS_H_
#define _MACROS_H_

#include <stdint.h>

#define CUMP_PREFETCH_FROM_CUDA_ARRAY 0 // use 2D texturing
#define CUMP_USE_PAGELOCKED_MEM       1 // allocate page-locked mem
#define CUMP_USE_PTX_ASSEMBLY         1 // use PTX assembly instead of
                                        // intrinsics
#define CUMP_USE_ATOMICS              1 // use atomic intructions
#define CUMP_VERBOSE  1

#define CUMP_BENCHMARK_CALLS 0  // benchmark separate kernels
#define CUMP_MEASURE_MEM 1      // measure time for GPU-host memory transfer

#define CUMP_COMPILE_DEBUG_KERNEL      1

#if CUMP_VERBOSE
#define STILL_ALIVE printf("%d\n", __LINE__)
#define CUMP_out(x) std::cerr << x
#define CUMP_out2(fmt, ...) printf(fmt, ##__VA_ARGS__)
#define XPRINTZ(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#else
#define STILL_ALIVE
#define CUMP_out(x)
#define CUMP_out2(x, ...)
#define XPRINTZ(fmt, ...)

#endif

#define WS 32
#define HF 16
#define LOG_WARP_SIZE 5

#define CLASS_NO_COPY(Type)      \
    Type(const Type &) = delete; \
    Type &operator=(const Type &) = delete

#define CU_SYNC __syncthreads();

#define CU_MEMFENCE __threadfence();

#define CUMP_LAUNCH_BOUNDS(t, b) __launch_bounds__(t, b)

// computes r = a - b subop c unsigned using extended precision
#define VSUBx(r, a, b, c, subop) \
     asm volatile("vsub.u32.u32.u32." subop " %0, %1, %2, %3;" :  \
                "=r"(r) : "r"(a) , "r"(b), "r"(c));

// computes r = a + b subop c unsigned using extended precision
#define VADDx(r, a, b, c, subop) \
     asm volatile("vadd.u32.u32.u32." subop " %0, %1, %2, %3;" :  \
                "=r"(r) : "r"(a) , "r"(b), "r"(c));

#define CUMP_THROW(x) throw x();

#define CUMP_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#define CUMP_SAFE_CALL_drv( call) {                                    \
    CUresult err = call;                                                    \
    if( CUDA_SUCCESS != err) {                                                \
        fprintf(stderr, "Cuda drv error in file '%s' in line %i: %d.\n",    \
                __FILE__, __LINE__, err);              \
        exit(EXIT_FAILURE);                                                  \
    } }


#define CU_CHECK_ERROR(err)         \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error %s:%i : %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(err) );               \
    }

#define CU_SAVEn(addr, idx, ofs, a, _n) do { \
    for(unsigned _i = 0; _i < _n; _i++) { \
        (addr)[(idx) + _i * (ofs)] = (a)[_i]; \
    }  } while(0);

#define CU_LOADn(addr, idx, ofs, a, _n) do { \
    for(unsigned _i = 0; _i < _n; _i++) { \
        (a)[_i] = (addr)[(idx) + _i * (ofs)]; \
    }  } while(0);

#define CU_LOAD_GLOBALn_dummy(addr, idx, ofs, a, _n) do { \
    for(unsigned _i = 0; _i < _n; _i++) { \
        (a)[_i] = (limb&)(addr) + _i; \
    }  } while(0);

#define CU_LOAD_GLOBALn(addr, idx, ofs, a, _n) do { \
    for(unsigned _i = 0; _i < _n; _i++) { \
        (a)[_i] = (addr)[(idx) + _i * (ofs)]; \
    }  } while(0);

#define CPU_BEGIN_TIMING(ID) \
        auto z1_##ID = std::chrono::high_resolution_clock::now();

#define CPU_END_TIMING(ID)                                                    \
        auto z2_##ID = std::chrono::high_resolution_clock::now();              \
        std::chrono::duration<double, std::milli> ms_##ID = z2_##ID - z1_##ID; \
        fprintf(stderr, "\n---------------------- %s time elapsed: %f msec\n", #ID, ms_##ID.count())

#define CU_BEGIN_TIMING(N_ITERS) { \
    cudaDeviceSynchronize();       \
    uint32_t nIters = N_ITERS;  \
    for(unsigned i = 0; i < nIters + 1; i++) {

#define CU_END_TIMING( name ) \
        if(i == 0) { \
            cudaDeviceSynchronize(); \
            cudaEventRecord( e_start, 0 ); \
        } \
    } \
    cudaEventRecord( e_end, 0 ); \
    cudaEventSynchronize( e_end ); \
    float ms = 0;                  \
    cudaEventElapsedTime( &ms, e_start, e_end ); \
    if(nIters > 0) ms /= nIters; \
    fprintf(stderr, "%s time elapsed: %.3f ms..\n", #name, ms); \
    }

#define FETCH_FROM_CUDA_ARRAY
#ifdef FETCH_FROM_CUDA_ARRAY
#define TEX1D(tex, index) tex1D((tex), (index))
#define TEX2D(tex, x, y)  tex2D((tex), (x), (y))
#else
#define TEX1D(tex, index) tex1Dfetch((tex), (index))
#define TEX2D(tex, x, y) (void)0
#endif

#if CUMP_PREFETCH_FROM_CUDA_ARRAY
#define TEX_ROW_FETCH(_a, _, _lin_ofs, _stride) do { \
        unsigned _ofs = _lin_ofs + _stride, \
            _row_ofs = _ofs & (TILE_SZ*TILE_SZ*N_TILES_PER_ROW - 1), \
            _lidx = _row_ofs & TILE_SZ*TILE_SZ - 1, \
            _lx = __umul24(_row_ofs >> 2*LOG_TILE_SZ, 0x1000000 + TILE_SZ) + \
                (_lidx & TILE_SZ - 1), _ly; \
        _row_ofs = _ofs / (TILE_SZ*TILE_SZ*N_TILES_PER_ROW); \
        _ly = __umul24(_row_ofs, 0x1000000 + TILE_SZ) + (_lidx / TILE_SZ); \
        (_a) = TEX2D(tiled_tex, _lx, _ly); \
    } while(0);

#define TEX_COLUMN_FETCH(_a, _, _lin_ofs, _stride) do { \
        unsigned _ofs = _lin_ofs + _stride, \
        _col_ofs = _ofs / (TILE_SZ*TILE_SZ*N_TILES_PER_COL), \
        _lx = __umul24(_col_ofs, 0x1000000 + TILE_SZ) + \
            (_ofs & TILE_SZ - 1), _ly; \
        _col_ofs = _ofs & (TILE_SZ*TILE_SZ*N_TILES_PER_COL - 1), \
        _ly = _col_ofs / TILE_SZ; \
        (_a) = TEX2D(tiled_tex, _lx, _ly); \
    } while(0);

#else
#define TEX_ROW_FETCH(_a, _g_in, _lin_ofs, _stride) do { \
        (_a) = (_g_in)[_lin_ofs + _stride]; \
    } while(0);

#define TEX_COLUMN_FETCH(_a, _g_in, _lin_ofs, _stride) \
        TEX_ROW_FETCH(_a, _g_in, _lin_ofs, _stride)

#endif 

#endif // _MACROS_H_
