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
    
#include <ctime>
#include <array>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "include/playground_host.h"

GPU_playground::GPU_playground()
{
    device_static_setup(CL_DEVICE_TYPE_GPU);
}

GPU_playground::~GPU_playground() {
    free_device_mem();
}

void GPU_playground::device_static_setup(uint32_t devType) {

    cl::vector<cl::Platform> platforms;  
    cl::Platform::get(&platforms);  

    std::cout << "Number of Platforms: " << platforms.size() << std::endl;

    bool found = false;
    for(const auto& platform : platforms) {
        auto name = platform.getInfo<CL_PLATFORM_NAME>();
        std::cout << "Platform Name: " << name << std::endl;  
        std::cout << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;  

        cl::vector<cl::Device> devices;  
        platform.getDevices(devType, &devices);  

        for(const auto& device : devices) {

#define printDev(what, ID) std::cout << "\t\t" << what << device.getInfo<ID>() << std::endl;
            printDev("Device Name: ", CL_DEVICE_NAME);
            printDev("Device Type: ", CL_DEVICE_TYPE);
            std::cout << "\t\tGPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;  
            printDev("Device Vendor: ", CL_DEVICE_VENDOR);
            printDev("Device Max Compute Units: ", CL_DEVICE_MAX_COMPUTE_UNITS);
            printDev("Device Global Memory: ", CL_DEVICE_GLOBAL_MEM_SIZE);
            printDev("Device Max Clock Frequency: ", CL_DEVICE_MAX_CLOCK_FREQUENCY);
            printDev("Device Max Allocateable Memory: ", CL_DEVICE_MAX_MEM_ALLOC_SIZE);
            printDev("Device Local Memory: ", CL_DEVICE_LOCAL_MEM_SIZE);
            printDev("Device Available: ",  CL_DEVICE_AVAILABLE);
            printDev("Device version: ", CL_DEVICE_VERSION);
            printDev("Driver version: ", CL_DRIVER_VERSION);
            printDev("OpenCL version: ", CL_DEVICE_OPENCL_C_VERSION);
#undef printDev
        }  
        if (name.find("HD Graphics") != std::string::npos) {
            auto ret = cl::Platform::setDefault(platform);
            if (ret != platform)
                throw std::runtime_error("Error setting default platform!");
            std::cout << "Setting as default platform!\n";
            found = true;

            std::cout << "Default device: " << cl::Device::getDefault().getInfo<CL_DEVICE_NAME>();
        }
        std::cout << std::endl;
    } 
}

//! \c CPU_mem_ptr collects all dynamic memory allocated either in ordinary
//! or page-locked memory triggered by the flag \c CUMP_USE_PAGELOCKED_MEM
//! \c DEV_mem_ptr collects all dynamic memory allocated on GPU, optionally
//! input operands can be allocated in texture space pointed to by
//! \c batch_size - # of inputs to allocate memory required for
//! parallel processing
void GPU_playground::alloc_device_mem() {

    free_device_mem(); // cleanup previously allocated memory
#if 0
    if(dev_mem_size + dev_mem_size_aux != 0) {
        OCL_SAFE_CALL(cudaMalloc(&DEV_mem_ptr, (dev_mem_size +
                dev_mem_size_aux) * word_size));

        if(DEV_mem_ptr == 0) {
            printf("ERROR: unable to allocate device mem..\n");
            exit(1);
        }
    }

    if(pinned_mem_size != 0) {// allocates device-mapped memory
        // cudaHostAllocWriteCombined ??
        OCL_SAFE_CALL(cudaHostAlloc(&CPU_pinned_mem_ptr, pinned_mem_size *
             word_size, cudaHostAllocMapped));

        OCL_SAFE_CALL(cudaHostGetDevicePointer(&DEV_pinned_mem_ptr,
              CPU_pinned_mem_ptr, 0));
    }

    if(pinned_mem_size != 0 &&
            (CPU_pinned_mem_ptr == 0 || DEV_pinned_mem_ptr == 0)) {
        printf("ERROR: unable to allocate device-mapped host mem..\n");
        exit(1);
    }

    if(CPU_mem_size + CPU_mem_size_aux != 0) {
        OCL_SAFE_CALL(cudaMallocHost(&CPU_mem_ptr, (CPU_mem_size +
            CPU_mem_size_aux) * word_size));

        if(CPU_mem_ptr == 0) {
            printf("ERROR: unable to allocate CPU mem..\n");
            exit(1);
        }
    }
#endif
}

void GPU_playground::free_device_mem() {

#if 0
    if(CPU_mem_ptr != 0) {
        printf("freeing page-locked mem..\n");
        OCL_SAFE_CALL(cudaFreeHost(CPU_mem_ptr));
    }

    if(CPU_pinned_mem_ptr != 0) {
        printf("freeing device-mapped host mem..\n");
        OCL_SAFE_CALL(cudaFreeHost(CPU_pinned_mem_ptr));
    }
    if(DEV_mem_ptr != 0) {
        printf("freeing device mem..\n");
        OCL_SAFE_CALL(cudaFree(DEV_mem_ptr));
    }

#if CUMP_PREFETCH_FROM_CUDA_ARRAY
    if(g_tiled_tex_array != 0) {
        CUMP_out("freeing CUDA array..\n");
        OCL_SAFE_CALL(cudaFreeArray(g_tiled_tex_array));
    }
#endif
#endif
}

FloatMatrix genIdentity(size_t w) {

    FloatMatrix C(w, w, 0.0);
    for(size_t y = 0; y < w; y++) {
        C[y][y] = 1;
    }
    return C;
}

FloatMatrix genConst(size_t w, size_t h, const FloatMatrix::NT& val) {

    FloatMatrix C(w, h, val);
    return C;
}

FloatMatrix genRandomMatrix(size_t w, size_t h) {

    FloatMatrix C(w, h);
    for(size_t y = 0; y < h; y++) {
        auto pC = C[y];
        for(size_t x = 0; x < w; x++) {
            pC[x] = FloatMatrix::NT(rand() / 1000.0);
        }
    }
    return C;
}

FloatMatrix matrixMulHost(const FloatMatrix& A, const FloatMatrix& B) {

    const auto w1 = A.width(), h1 = A.height(),
              w2 = B.width(), h2 = B.height(), stepB = B.stepT();

    if(w1 != h2) {
        throw std::runtime_error("Input matrix dimensions do not agree!");
    }

    FloatMatrix C(w2, h1);
    for(size_t y = 0; y < h1; y++) {

        auto pA = A[y];
        auto pC = C[y];
        for(size_t x = 0; x < w2; x++) {

            auto pB = B.data() + x;
            FloatMatrix::NT sum(0.0);
            for(size_t i = 0; i < w1; i++) { // w1 == h2
                sum += pA[i] * pB[i*stepB];
            }
            pC[x] = sum;
        }
    }
    return C;
}

void GPU_playground::generate_data() {

    srand((uint32_t)time(NULL));
    size_t hA = 100, wB = 100, D = 100;
    // TODO: next step: make the A.w = B.h larger than 16:
#if 1
    m_A = genRandomMatrix(D, hA);
    m_B = genRandomMatrix(wB, D);
#else
    m_A = genConst(D, hA, 1);
    m_B = genConst(wB, D, 1);
#endif
    m_C = matrixMulHost(m_A, m_B);
}

void GPU_playground::runMatrixMul(cl::Program& program)
{
    XPRINTZ("generating data begin..");
    generate_data();
    XPRINTZ("generating data end..");

    word_size = sizeof(Real);
    auto szA = m_A.size(), szB = m_B.size(),
         szC = m_C.size();

    dev_mem_size = szA + szB;
    pinned_mem_size = szC;
    alloc_device_mem();
     
    const size_t N = 1000;
    std::vector< float > vecC(m_C.size(), 777.0f);

    cl::Buffer devA(begin(m_A), end(m_A), true, true); // readonly, useHostPtr
    cl::Buffer devB(begin(m_B), end(m_B), true, true); // readonly, useHostPtr
    cl::Buffer devC(begin(m_C), end(m_C), false); // readonly, useHostPtr

    cl_int error;
    auto vecAddKernel = cl::KernelFunctor<
            cl::Buffer&,
            cl::Buffer&,
            cl::Buffer&,
            uint32_t
            >(program, "vectorAdd");  

    auto evt = vecAddKernel(cl::EnqueueArgs{
            cl::NDRange{ m_C.size() }
        },
        devA, devB, devC, m_C.size(), error);

    evt.wait();
    std::cout << "Execution: " << getErrorStr(error) << std::endl;

    auto device = cl::Device::getDefault();
    
    //auto [X,Y,Z] = vecAddKernel.getKernel().getWorkGroupInfo<CL_KERNEL_GLOBAL_WORK_SIZE>(device);
    //std::cout << "Array return: " << X << ", " << Y << ", " << Z << "\n";

    auto W = vecAddKernel.getKernel().getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
    std::cout << "Work group size: " << W << "\n";

    cl::copy(devC, begin(vecC), end(vecC));
    for (size_t i = 0; i < vecC.size(); i++) {
        XPRINTZ("%d: %f", i, vecC[i]);
    }

#if 0
    auto outC = (Real *)CPU_pinned_mem_ptr;
    std::fill(outC, outC + szC, Real(1111));

   // launch_matmul();
    checkme(outC, m_C.data(), m_C.width(), m_C.stepT(), m_C.height(), 1e-1f, true);

    XPRINTZ("C matrix size: %d x %d; stride: %d", m_C.width(), m_C.height(),
            m_C.stepT());
#endif
}

void GPU_playground::runReduce() {

    word_size = sizeof(ReduceType);

    std::vector< ReduceType > data(256),
            result(data.size());

    srand((uint32_t)time(NULL));

    ReduceType sum = 0;
    for(size_t i = 0; i < data.size(); i++) {
        data[i] = rand();
        sum += data[i];
        result[i] = sum;
    }

    pinned_mem_size = data.size() + result.size();
    alloc_device_mem();

    auto in = (ReduceType *)CPU_pinned_mem_ptr,
            out = in + data.size();

    std::copy(data.begin(), data.end(), in);
    std::fill(out, out + result.size(), ReduceType(1111));

   // launch_reduce(data.size());
    checkme(out, result.data(), result.size(), result.size(), 1, ReduceType(0), false);
}

bool GPU_playground::run(int argc, char **argv) {

   // Read the program source
     std::string fpath(__FILE__); // search for file in the source directory
     auto pos = fpath.find_last_of("\\/");
     fpath = fpath.substr(0, pos+1);
     std::ifstream ifs(fpath + "playground_gpu.cl");
     if (!ifs.is_open())
         throw std::runtime_error("Unable to open source CL file for reading!");

    std::string source(std::istreambuf_iterator{ifs}, {});

    XPRINTZ("Compiling source code..");
    cl::Program program(source, false);

    try {
        program.build("-cl-std=CL2.0");
    }
    catch (...) {
        std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault()) 
                  << std::endl;
        return false;
    }
    auto vec = program.getInfo< CL_PROGRAM_BINARIES >();
    for (const auto& block : vec) {
        std::cout << "Program binary size: " << block.size() << std::endl;
    }

    //runReduce();
    //runMatrixMul(program);
    return true;
}

#define CaseReturnString(x) case x: return #x;

const char *GPU_playground::getErrorStr(int err)
{
    switch (err)
    {
        CaseReturnString(CL_SUCCESS                        )                                  
        CaseReturnString(CL_DEVICE_NOT_FOUND               )
        CaseReturnString(CL_DEVICE_NOT_AVAILABLE           )
        CaseReturnString(CL_COMPILER_NOT_AVAILABLE         ) 
        CaseReturnString(CL_MEM_OBJECT_ALLOCATION_FAILURE  )
        CaseReturnString(CL_OUT_OF_RESOURCES               )
        CaseReturnString(CL_OUT_OF_HOST_MEMORY             )
        CaseReturnString(CL_PROFILING_INFO_NOT_AVAILABLE   )
        CaseReturnString(CL_MEM_COPY_OVERLAP               )
        CaseReturnString(CL_IMAGE_FORMAT_MISMATCH          )
        CaseReturnString(CL_IMAGE_FORMAT_NOT_SUPPORTED     )
        CaseReturnString(CL_BUILD_PROGRAM_FAILURE          )
        CaseReturnString(CL_MAP_FAILURE                    )
        CaseReturnString(CL_MISALIGNED_SUB_BUFFER_OFFSET   )
        CaseReturnString(CL_COMPILE_PROGRAM_FAILURE        )
        CaseReturnString(CL_LINKER_NOT_AVAILABLE           )
        CaseReturnString(CL_LINK_PROGRAM_FAILURE           )
        CaseReturnString(CL_DEVICE_PARTITION_FAILED        )
        CaseReturnString(CL_KERNEL_ARG_INFO_NOT_AVAILABLE  )
        CaseReturnString(CL_INVALID_VALUE                  )
        CaseReturnString(CL_INVALID_DEVICE_TYPE            )
        CaseReturnString(CL_INVALID_PLATFORM               )
        CaseReturnString(CL_INVALID_DEVICE                 )
        CaseReturnString(CL_INVALID_CONTEXT                )
        CaseReturnString(CL_INVALID_QUEUE_PROPERTIES       )
        CaseReturnString(CL_INVALID_COMMAND_QUEUE          )
        CaseReturnString(CL_INVALID_HOST_PTR               )
        CaseReturnString(CL_INVALID_MEM_OBJECT             )
        CaseReturnString(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        CaseReturnString(CL_INVALID_IMAGE_SIZE             )
        CaseReturnString(CL_INVALID_SAMPLER                )
        CaseReturnString(CL_INVALID_BINARY                 )
        CaseReturnString(CL_INVALID_BUILD_OPTIONS          )
        CaseReturnString(CL_INVALID_PROGRAM                )
        CaseReturnString(CL_INVALID_PROGRAM_EXECUTABLE     )
        CaseReturnString(CL_INVALID_KERNEL_NAME            )
        CaseReturnString(CL_INVALID_KERNEL_DEFINITION      )
        CaseReturnString(CL_INVALID_KERNEL                 )
        CaseReturnString(CL_INVALID_ARG_INDEX              )
        CaseReturnString(CL_INVALID_ARG_VALUE              )
        CaseReturnString(CL_INVALID_ARG_SIZE               )
        CaseReturnString(CL_INVALID_KERNEL_ARGS            )
        CaseReturnString(CL_INVALID_WORK_DIMENSION         )
        CaseReturnString(CL_INVALID_WORK_GROUP_SIZE        )
        CaseReturnString(CL_INVALID_WORK_ITEM_SIZE         )
        CaseReturnString(CL_INVALID_GLOBAL_OFFSET          )
        CaseReturnString(CL_INVALID_EVENT_WAIT_LIST        )
        CaseReturnString(CL_INVALID_EVENT                  )
        CaseReturnString(CL_INVALID_OPERATION              )
        CaseReturnString(CL_INVALID_GL_OBJECT              )
        CaseReturnString(CL_INVALID_BUFFER_SIZE            )
        CaseReturnString(CL_INVALID_MIP_LEVEL              )
        CaseReturnString(CL_INVALID_GLOBAL_WORK_SIZE       )
        CaseReturnString(CL_INVALID_PROPERTY               )
        CaseReturnString(CL_INVALID_IMAGE_DESCRIPTOR       )
        CaseReturnString(CL_INVALID_COMPILER_OPTIONS       )
        CaseReturnString(CL_INVALID_LINKER_OPTIONS         )
        CaseReturnString(CL_INVALID_DEVICE_PARTITION_COUNT )
        default: return "Unknown OpenCL error code";
    }
}
