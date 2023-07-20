    
#include <iostream>
#include <ctime>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <random>

#include "playground_host.h"
#include "interpolate_host.h"
#include "display_images_CV.h"

GPU_playground::GPU_playground()
{
    device_static_setup();
}

GPU_playground::~GPU_playground()
{
    free_device_mem();
    cudaEventDestroy(e_start);
    cudaEventDestroy(e_end);
}

void GPU_playground::device_static_setup() {

    int cnt;
    CUMP_SAFE_CALL(cudaGetDeviceCount(&cnt));
    if(cnt == 0) {
        XPRINTZ("\nSorry, your graphics card does not support CUDA, "
            "do you want to play Solitaire instead ?");
        exit(1);
    }

    int dev = 0;
    cudaDeviceProp props;
    CUMP_SAFE_CALL(cudaGetDeviceProperties(&props, dev));
    if(props.major < 1) {
        XPRINTZ("Device does not support CUDA");
        exit(1);
    }
    XPRINTZ("\nFound CUDA compatible device: %s; SM version: %d.%d"
           "\nTotal device mem: %f Mb", props.name, props.major,
            props.minor, (double)props.totalGlobalMem / (1024*1024));

    XPRINTZ("\nsharedMemPerBlock = %llu"
           "\nregsPerBlock = %d"
           "\nwarpSize = %d"
           "\nmemPitch = %llu"
           "\nmaxThreadsPerBlock = %d",
            props.sharedMemPerBlock,
            props.regsPerBlock,
            props.warpSize,
            props.memPitch,
            props.maxThreadsPerBlock);

    CUMP_SAFE_CALL(cudaSetDevice(dev));

    cudaEventCreate(&e_start);
    cudaEventCreate(&e_end);
}

//! \c CPU_mem_ptr collects all dynamic memory allocated either in ordinary
//! or page-locked memory triggered by the flag \c CUMP_USE_PAGELOCKED_MEM
//! \c DEV_mem_ptr collects all dynamic memory allocated on GPU, optionally
//! input operands can be allocated in texture space pointed to by
//! \c batch_size - # of inputs to allocate memory required for
//! parallel processing
void GPU_playground::alloc_device_mem() {

    free_device_mem(); // cleanup previously allocated memory

    if(dev_mem_size != 0) {
        CUMP_SAFE_CALL(cudaMalloc(&DEV_mem_ptr, dev_mem_size * word_size));

        if(DEV_mem_ptr == 0) {
            XPRINTZ("ERROR: unable to allocate device mem..");
            exit(1);
        }
        XPRINTZ("Allocated %zu bytes of device memory", dev_mem_size * word_size);
    }

    if(pinned_mem_size != 0) {// allocates device-mapped memory
        // cudaHostAllocWriteCombined ??
        CUMP_SAFE_CALL(cudaHostAlloc(&CPU_pinned_mem_ptr, pinned_mem_size *
             word_size, cudaHostAllocMapped));

        CUMP_SAFE_CALL(cudaHostGetDevicePointer(&DEV_pinned_mem_ptr,
              CPU_pinned_mem_ptr, 0));

        // CPU_pinned_mem_ptr == DEV_pinned_mem_ptr for 64 bits
        XPRINTZ("Allocated %zu bytes of pinned device memory",
                pinned_mem_size * word_size);
    }

    if(pinned_mem_size != 0 &&
            (CPU_pinned_mem_ptr == 0 || DEV_pinned_mem_ptr == 0)) {
        XPRINTZ("ERROR: unable to allocate device-mapped host mem..");
        exit(1);
    }

    if(CPU_mem_size != 0) {
        CUMP_SAFE_CALL(cudaHostAlloc(&CPU_mem_ptr, CPU_mem_size * word_size, cudaHostAllocDefault));

        if(CPU_mem_ptr == 0) {
            XPRINTZ("ERROR: unable to allocate CPU mem..");
            exit(1);
        }
        XPRINTZ("Allocated %zu bytes of page-locked host memory", CPU_mem_size * word_size);
    }
}

void GPU_playground::free_device_mem() {

    if(CPU_mem_ptr != 0) {
        XPRINTZ("freeing page-locked mem..");
        CUMP_SAFE_CALL(cudaFreeHost(CPU_mem_ptr));
    }

    if(CPU_pinned_mem_ptr != 0) {
        XPRINTZ("freeing device-mapped host mem..");
        CUMP_SAFE_CALL(cudaFreeHost(CPU_pinned_mem_ptr));
    }
    if(DEV_mem_ptr != 0) {
        XPRINTZ("freeing device mem..");
        CUMP_SAFE_CALL(cudaFree(DEV_mem_ptr));
    }

#if CUMP_PREFETCH_FROM_CUDA_ARRAY
    if(g_tiled_tex_array != 0) {
        XPRINTZ("freeing CUDA array..");
        CUMP_SAFE_CALL(cudaFreeArray(g_tiled_tex_array));
    }
#endif
}

static FloatMatrix genIdentity(size_t w) {

    FloatMatrix C(w, w, 0.0);
    for(size_t y = 0; y < w; y++) {
        C[y][y] = 1;
    }
    return C;
}

static FloatMatrix genConst(size_t w, size_t h, const FloatMatrix::NT& val) {
    return FloatMatrix(w, h, val);
}

static FloatMatrix genRandomMatrix(size_t w, size_t h) {

    FloatMatrix C(w, h);
    for(size_t y = 0; y < h; y++) {
        auto pC = C[y];
        for(size_t x = 0; x < w; x++) {
            pC[x] = FloatMatrix::NT(rand() / 1000.0);
        }
    }
    return C;
}

static FloatMatrix matrixMulHost(const FloatMatrix& A, const FloatMatrix& B) {

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

void GPU_matrixMul::generate_data() {

    srand((uint32_t)time(NULL));
    size_t hA = 16*5+9, wB = 16+1, D = 16*7+13;
    // TODO: next step: make the A.w = B.h larger than 16:
#if 0
    m_A = genRandomMatrix(D, hA);
    m_B = genRandomMatrix(wB, D);
#else
    m_A = genConst(D, hA, 2);
    //m_B = genConst(wB, D, 3);
    m_B = genRandomMatrix(wB, D);

#endif
    m_C = matrixMulHost(m_A, m_B);
}

void GPU_matrixMul::run() {

    XPRINTZ("generating data begin..");
    generate_data();
    XPRINTZ("generating data end..");

    word_size = sizeof(Real);
    auto szA = m_A.size(), szB = m_B.size(),
         szC = m_C.size();

    XPRINTZ("C: %u; (%u x %u) stride: %u -- %u",
            (uint32_t)m_C.size(), (uint32_t)m_C.width(), (uint32_t)m_C.height(),
            (uint32_t)m_C.stepT(), (uint32_t)m_B.stepT());

    dev_mem_size = szA + szB;
    pinned_mem_size = szC;
    alloc_device_mem();

    auto outC = (Real *)CPU_pinned_mem_ptr;
    std::fill(outC, outC + szC, Real(1111));

    launchKernel();
    checkme(outC, m_C.data(), m_C.width(), m_C.stepT(), m_C.height(), 1e-1f, true);

    XPRINTZ("C matrix size: %u x %u; stride: %u",
            (uint32_t)m_C.width(), (uint32_t)m_C.height(), (uint32_t)m_C.stepT());
}

void GPU_reduce::run() {

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

    launchKernel(data.size());
    checkme(out, result.data(), result.size(), result.size(), 1, ReduceType(0), false);
}

template < class InputReal >
static std::tuple<size_t, std::vector< InputReal >> generateGaussian(int w, int h, size_t nDataSigs)
{
    cv::Mat orig(cv::Size(w, h), CV_32F, cv::Scalar(0));
    const auto s_nan = std::numeric_limits< float >::quiet_NaN();

    int ofs = 0;
    float sigmaX = 150, sigmaY = 190, x0 = w*0.5f, y0 = h*0.5f;

    for(int y = ofs; y < h - ofs; y++) {
        auto pin = orig.ptr< float >(y);
        float dy = (y - y0) / sigmaY;
        dy *= dy;

        for(int x = ofs; x < w - ofs; x++) {
            float dx = (x - x0) / sigmaX;
            dx *= dx;
            float z = std::abs(10*std::exp(-dx-dy));
            pin[x] = (z < 1e-3 ? s_nan : z);
        }
    }

    srand((uint32_t)time(nullptr));
    for(int i = 0; i < 50; i++) {

        double x = w*(double)rand() / (RAND_MAX+1),
               y = h*(double)rand() / (RAND_MAX+1),
               r = 10 * (double)rand() / (RAND_MAX+1);
        cv::circle(orig, cv::Point2d(x,y), (int)r, cv::Scalar(s_nan), -1);
    }

    size_t bufS = w * h;
    std::vector< InputReal > out(bufS * (nDataSigs + 2));
    auto pX = out.data(), pY = pX + bufS, pData = pY + bufS;

    for(int y = 0, i = 0; y < h; y++) {
        auto pin = orig.ptr< float >(y);
        for(int x = 0; x < w; x++, i++) {
            pX[i] = (InputReal)x;
            pY[i] = (InputReal)y;
            auto pp = pData;
            for(size_t j = 0; j < nDataSigs; j++, pp += bufS) {
                pp[i] = (InputReal)pin[x];
            }
        }
    }
    return { w*h, out };
}

uint32_t popcnt(uint32_t val) {
    uint32_t num = 0;
    while(val) {
        if(val & 1)
            num++;
        val >>= 1;
    }
    return num;
}

// 0-based position of the Nth LSB set
uint32_t getNthBitPos(uint32_t val, uint32_t N) {

    uint32_t pos = 0;
    for(uint32_t ofs = 16; ofs != 0; ofs /= 2)
    {
        auto mval = val & ((1 << ofs)-1);
        auto dif = (int)(N - popcnt(mval));

//        XPRINTZ("ofs = %d; mval: 0x%X; pos: %d; val: 0x%X; N: %d; dif: %d",
//                ofs, mval, pos, val, N, dif);

        if(dif <= 0) {
            val = mval;
        } else {
            N = dif;
            pos += ofs;
            val >>= ofs;
        }
    }
    XPRINTZ("last N = %d; lastVAL; %d", N, val);
    return val == 1 && N == 1 ? pos+1 : 0;
}

void GPU_interpolator::run() {

    using DataInterp = DataInterpolator<InputReal, OutputReal>;
    DataInterp zobj;

    const uint32_t nCols = 500, nRows = 500,
            numDataSigs = 1;
    const float stepX = 1, stepY = 1,
          innerRad = 3, outerRad = 5;

//    uint32_t num = 0x55555555;
//    for(int N = 1; N <= 32; N++) {
//        XPRINTZ("number: 0x%X; pocnt: %d; %d-th bitpos: %d", num, popcnt(num), N,
//            getNthBitPos(num, N));
//    }
//    return;

    DataInterp::Params p = {
        nCols,
        nRows,
        numDataSigs,
        innerRad,
        innerRad * stepY / stepX,
        // limit the maximal pixel radius for point search..
        outerRad,
        outerRad * stepY / stepX,
    };
    p.innerRadX = std::min(p.innerRadX, p.outerRadX);
    p.innerRadY = std::min(p.innerRadY, p.outerRadY);

    XPRINTZ("stepXY: (%.3f;%.3f) pix/mm; innerRad: (%.2f;%.2f) pix; outerRad: (%.2f;%.2f) pix",
            stepX, stepY, p.innerRadX, p.innerRadY, p.outerRadX, p.outerRadY);

    cv::Mat outImg;
    if constexpr(std::is_same_v< OutputReal, float >)
        outImg = cv::Mat(cv::Size(p.w, p.h), CV_32FC(p.numSignals));
    else if constexpr(std::is_same_v< OutputReal, double >)
        outImg = cv::Mat(cv::Size(p.w, p.h), CV_64FC(p.numSignals));

    zobj.setup(outImg.ptr< OutputReal >(), p);

    std::string fpath(__FILE__); // search for file in the source directory
    fpath = fpath.substr(0, fpath.find_last_of("\\/")); // what if npos ??

    // this returns 'numDataSigs + 2' arrays of type InputReal each of size 'nCols * nRows'
    const auto [nSamples, in] = //zobj.readFromFile(fpath + "/output_20Khz_new_500x500.bin");
                               generateGaussian< InputReal >(nCols, nRows, numDataSigs);
    m_nSamples = nSamples;
    word_size = 1;
    // input and output buffers are pinned memory
    pinned_mem_size = in.size() * sizeof(InputReal) + outImg.total() * p.numSignals * sizeof(OutputReal);
    // internal device buffer for collecting samples..
    dev_mem_size = p.w * p.h * p.numSignals * sizeof(DataInterp::Pix);

    XPRINTZ("------- outImg total: %zu", outImg.total());

    alloc_device_mem();
    m_devIn = (InputReal *)DEV_pinned_mem_ptr;
    m_devOut = (OutputReal *)(m_devIn + m_nSamples * (p.numSignals + 2));
    m_devOutCPU = m_devOut; // same address space..

    memcpy(m_devIn, in.data(), in.size() * sizeof(InputReal));

    launchKernel(p);

    std::atomic_bool cancel{};
    CPU_BEGIN_TIMING(HostInterp);

    //for(auto start = in.data(), ptr = start; ptr < start + in.size(); ptr += numDataSigs + 2) {
    std::vector< InputReal > xbuf(numDataSigs + 2); // temporary buf for SoA -> AoS
    for(uint32_t i = 0; i < m_nSamples; i++) {

        auto pp = in.data() + i;
        for(uint32_t k = 0, sz = p.w * p.h; k < numDataSigs + 2; k++, pp += sz) {
            xbuf[k] = pp[0];
        }
        zobj.addPoint(xbuf.data());
    }
    zobj.process(cancel);
    CPU_END_TIMING(HostInterp);

    bool print_if_differs = true;
    checkme(m_devOutCPU, outImg.ptr< OutputReal >(), p.w, p.w, p.h,
            (OutputReal)1e-3, print_if_differs, 1000);

    DisplayImageCV disp;
    //disp.show(cv::Mat(cv::Size(p.w, p.h), CV_32FC(p.numSignals), m_devOutCPU));
    disp.show(outImg);
}

// need vector type for memory allocations..

// algorithm to setting
void GPU_radixSort::run()
{
    word_size = 4;
    const uint32_t dataSize = 128*10;
    pinned_mem_size = dataSize;

    alloc_device_mem();

    m_cpuOut.resize(dataSize);
    m_pinnedData = (uint32_t *)DEV_pinned_mem_ptr;

    std::random_device rd; // this might be used to obtain seed
    std::mt19937 gen(111);
    std::uniform_int_distribution<uint32_t> uniform(0, (dataSize - 1)*32);

    uint32_t ssum = 0;
    for(uint32_t i = 0; i < dataSize; i++) {
        m_pinnedData[i] = uniform(gen) % 256;
        ssum += m_pinnedData[i];
        m_cpuOut[i] = ssum;
    }
    //std::copy(m_pinnedData, m_pinnedData + dataSize, begin(m_cpuOut));
   // std::sort(begin(m_cpuOut), end(m_cpuOut));

    launchKernel(dataSize);

    bool print_if_differs = true;
    checkme( (const int32_t *)m_pinnedData, (const int32_t *)m_cpuOut.data(), dataSize, dataSize, 1,
            0, print_if_differs, 1000);
}
