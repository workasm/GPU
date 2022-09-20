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

#ifndef GPU_PLAYGROUND_H
#define GPU_PLAYGROUND_H

#include <stdint.h>
#include <iostream>
#include <algorithm>
#include <vector>

#include "macros.h"

#define CL_API_SUFFIX__VERSION_2_2_DEPRECATED
#define CL_API_PREFIX__VERSION_2_2_DEPRECATED
#define CL_HPP_TARGET_OPENCL_VERSION 220
#define CL_HPP_ENABLE_EXCEPTIONS
#include "include/opencl.hpp"

template <typename T>
class TImage : std::vector< T > {

    enum {
        s_align = 4,
        s_ofs = (1 << s_align) - 1,
    };

    using Base = std::vector< T >;
public:
    using NT = T;
    using Base::data;
    using Base::size;
    using Base::begin;
    using Base::end;

    explicit TImage(size_t w = 0, size_t h = 0) {
        resize(w, h);
    }

    TImage(size_t w, size_t h, const T& value) {
        resize(w, h, value);
    }

    TImage(const T *ptr, size_t w, size_t h) :
        TImage(w, h)
    {
        std::copy(ptr, ptr + size(), data());
    }

    void resize(size_t w, size_t h) {
        m_width = w, m_height = h, m_stride = (w + s_ofs) & ~s_ofs;
        Base::resize(m_stride * h);
    }

    void resize(size_t w, size_t h, const T& value) {
        m_width = w, m_height = h, m_stride = (w + s_ofs) & ~s_ofs;
        Base::resize(m_stride * h, value);
    }

    void swap(TImage& rhs) {
        std::swap(m_width, rhs.m_width);
        std::swap(m_height, rhs.m_height);
        std::swap(m_stride, rhs.m_stride);
        Base::swap(rhs);
    }

    void fill(const T& value) {
        std::fill(begin(), end(), value);
    }

    const T* operator[](size_t i) const { return data() + i * m_stride; }
    T* operator[](size_t i)	{ return data() + i * m_stride; }

    size_t stepT() 	const { return m_stride; }          // line size in type T
    //size_t step() 	const { return stepT()*sizeof(T); } // line size in bytes

    size_t width() 	const { return m_width; }
    size_t height()	const { return m_height; }

    size_t sizeInBytes() const { return size() * sizeof(T); }

private:
     size_t m_width, m_height, m_stride;
};


//! compares 2D arrays of data, \c width elements per row stored with \c stride (stride == width)
//! number of rows given by \c n_batches
//! \c print_when_differs :  indicates whether print elements only if they
//! differ (default)
//! \c print_max : maximal # of entries to print
template < class NT >
bool checkme(const NT *checkit, const NT *truth, size_t width, size_t stride,
        size_t n_batches, const NT& eps, bool print_when_differs = true,
             size_t print_max = std::numeric_limits< size_t >::max()) {

    if(checkit == NULL)
        return false;

    bool res = true;
    size_t printed = 0;
    std::cerr << "\nlegend: batch_id (element_in_batch)\n";
    for(int j = (int)n_batches - 1; j >= 0; j--) {

        const NT *pcheckit = checkit + j * stride,
            *ptruth = truth + j * stride;
        for(int i = (int)width - 1; i >= 0; i--) {

            NT diff = pcheckit[i] - ptruth[i];
            bool differs = std::abs(diff) > eps;
            if(differs)
                res = false;

            if((differs || !print_when_differs) && printed < print_max) {
                NT check = pcheckit[i], truth = ptruth[i];

                printed++;
                std::cerr << j << '(' << i << ") (GPU, truth): " <<
                             check << " and " << truth << " ; diff: " << diff;
                if(differs)
                    std::cerr <<" DIFFERS\n";
                else
                    std::cerr << '\n';
            }
        }
    }
    return res;
}

using FloatMatrix = TImage< Real >;

struct DeviceMatrix {
    Real *data;
    uint32_t w, h, stride;
    Real get(int32_t x, int32_t y) const {
        return data[x + y * stride];
    }
};

struct GPU_playground {

    using ReduceType = int32_t;

    GPU_playground();
    ~GPU_playground();

    void alloc_device_mem();
    void free_device_mem();

    void device_static_setup(uint32_t type);
    void host_static_setup();

    void generate_data();
    bool run(int argc, char **argv);

    void runMatrixMul(cl::Program& program);
    void runReduce();

    static const char *getErrorStr(int err);

    //bool launch_matmul();
    //bool launch_reduce(size_t dataSz);

protected:
    FloatMatrix m_A, m_B, m_C;

    size_t word_size;
    size_t tm_rolex; // CUDA timer handle

    // device main & auxiliary mem sizes
    size_t dev_mem_size = 0, dev_mem_size_aux = 0;
    // host main & auxiliary mem sizes
    size_t CPU_mem_size = 0, CPU_mem_size_aux = 0;
    size_t pinned_mem_size = 0; // device-mapped memory size (on demand)

    void *CPU_mem_ptr = nullptr, *DEV_mem_ptr = nullptr; // host and device memory pointers
    void *CPU_pinned_mem_ptr = nullptr, *DEV_pinned_mem_ptr = nullptr; // pinned mem pointers
};

#endif // GPU_PLAYGROUND_H
