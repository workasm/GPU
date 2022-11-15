#ifndef GPU_PLAYGROUND_H
#define GPU_PLAYGROUND_H

#include <stdint.h>
#include <iostream>
#include <algorithm>
#include <vector>

#include "macros.h"
#include "common_types.h"

#include <cuda_runtime_api.h>

template <class T,
          std::enable_if_t< !std::is_floating_point_v<T>, bool> = true >
bool checkNaN(T) {
    return false;
}

template <class T,
     std::enable_if_t< std::is_floating_point_v<T>, bool> = true >
bool checkNaN(T a) {
    return std::isnan(a);
}


//! compares 2D arrays of data, \c width elements per row stored with \c stride (stride == width)
//! number of rows given by \c n_batches
//! \c print_when_differs :  indicates whether print elements only if they
//! differ (default)
//! \c print_max : maximal # of entries to print
template < bool Reverse = false, class NT >
bool checkme(const NT *checkit, const NT *truth, size_t width, size_t stride,
        size_t n_batches, const NT& eps, bool print_when_differs = true,
             size_t print_max = std::numeric_limits< size_t >::max()) {

    if(checkit == NULL)
        return false;

    bool res = true;
    size_t printed = 0;
    std::cerr << "\nlegend: batch_id (element_in_batch)\n";

    int inc = Reverse ? -1 : 1;
    size_t jbeg = 0, jend = n_batches,
           ibeg = 0, iend = width;

    if(Reverse) {
        jbeg = n_batches - 1, jend = (size_t)-1;
        ibeg = width - 1, iend = jend;
    }

    for(size_t j = jbeg; j != jend; j += inc) {

        const NT *pcheckit = checkit + j * stride,
            *ptruth = truth + j * stride;

        for(size_t i = ibeg; i != iend; i += inc) {

            if(checkNaN(ptruth[i]) && checkNaN(pcheckit[i]))
                continue;

            NT diff = pcheckit[i] - ptruth[i];
            bool isDiff = std::abs(diff) > eps;
            if(isDiff)
                res = false;

            if((isDiff || !print_when_differs) && printed < print_max) {
                NT check = pcheckit[i], truth = ptruth[i];

                printed++;
                std::cerr << j << '(' << i << ") (GPU, truth): " <<
                   check << " and " << truth << " ; diff: " << diff << (isDiff ? " DIFFERS\n" : "\n");
            }
        }
    }
    return res;
}

using FloatMatrix = TImage< float >;

struct DeviceMatrix {
    float *data;
    uint32_t w, h, stride;
    __device__ float get(int32_t x, int32_t y) const {
        return data[x + y * stride];
    }
};

struct GPU_playground {

    using ReduceType = int32_t;

    GPU_playground();
    virtual ~GPU_playground();

    void alloc_device_mem();
    void free_device_mem();

    void device_static_setup();
    void host_static_setup();

    virtual void run() = 0;
protected:

    size_t word_size;
    size_t tm_rolex; // CUDA timer handle

    // device & host mem sizes
    size_t dev_mem_size = 0, CPU_mem_size = 0;
    size_t pinned_mem_size = 0; // device-mapped memory size (on demand)

    void *CPU_mem_ptr = nullptr, *DEV_mem_ptr = nullptr; // host and device memory pointers
    void *CPU_pinned_mem_ptr = nullptr, *DEV_pinned_mem_ptr = nullptr; // pinned mem pointers

    cudaEvent_t e_start, e_end;
};

struct GPU_reduce : GPU_playground {

    ~GPU_reduce() override = default;

    void run() override;

protected:
    bool launchKernel(size_t dataSz);
};

struct GPU_matrixMul : GPU_playground {

    using Real = float;
    ~GPU_matrixMul() override = default;

    void run() override;

protected:
    bool launchKernel();
    void generate_data();

    FloatMatrix m_A, m_B, m_C;
};

struct GPU_interpolator : GPU_playground {

    using InputReal = double;   // input data & internal data type for interpolation
    using OutputReal = float;   // output data type

    GPU_interpolator() = default;
    ~GPU_interpolator() override = default;

    void run() override;
protected:
    bool launchKernel(const InterpParams< OutputReal >& params);

    size_t m_nSamples = 0;  // total number of data samples
    InputReal *m_devIn = nullptr;
    OutputReal *m_devOut = nullptr, *m_devOutCPU = nullptr;
};

#endif // GPU_PLAYGROUND_H
