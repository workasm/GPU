
#ifndef _MACROS_H_
#define _MACROS_H_

#include "include/flags.h"
#include <stdint.h>
#include <stdexcept>

using Real = float;

#define CUMP_VERBOSE 1

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

#define CU_SYNC __syncthreads();

#define CU_MEMFENCE __threadfence();

#define CUMP_LAUNCH_BOUNDS(t, b) __launch_bounds__(t, b)

#if 1 
#define OCL_SAFE_CALL(call) {                               \
    auto err = call;                                           \
    if(CL_SUCCESS != err) {                                   \
        throw std::runtime_error(std::to_string(__LINE__) + std::string(": OpenCL error: ") + getErrorStr(err)); \
    } }
#else
#define OCL_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#define OCL_CHECK_ERROR(errorMessage) do {                                 \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    err = cudaThreadSynchronize();                                           \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#define CL_BEGIN_TIMING(N_ITERS) { \
    cudaDeviceSynchronize();       \
    uint32_t __n_iters = N_ITERS;  \
    for(unsigned __iii = 0; __iii < __n_iters + 1; __iii++) {

#define CL_END_TIMING( ms ) \
        if(__iii == 0) { \
            cudaDeviceSynchronize(); \
            cudaEventRecord( e_start, 0 ); \
        } \
    } \
    cudaEventRecord( e_end, 0 ); \
    cudaEventSynchronize( e_end ); \
    cudaEventElapsedTime( &ms, e_start, e_end ); \
    ms /= __n_iters; \
    }
#endif

#endif // _MACROS_H_
