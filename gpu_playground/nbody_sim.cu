#include <iostream>
#include <fstream>
#include <math.h>

#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

//#ifndef __CUDACC__
#define __CUDACC__ 1
//#endif

// C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include/
#include <cooperative_groups.h>
//#include <sm_30_intrinsics.hpp>
using namespace cooperative_groups;

#include "macros.h"
#include "playground_host.h"
#include "common_funcs.cu"

#if 1
#define PRINTZ(fmt, ...) printf(fmt"\n", ##__VA_ARGS__)
#else
#define PRINTZ(fmt, ...)
#endif

  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>

double PCFreq = 0.0;
__int64 timerStart = 0;

void read_values_from_file(const char * file, float * data, size_t size)
{
    std::ifstream values(file, std::ios::binary);
    if(!values.is_open())
        throw std::runtime_error("Unable to read file " + std::string(file));

    values.read(reinterpret_cast<char*>(data), size);
    values.close();
}

void write_values_to_file(const char * file, float * data, size_t size) {
    std::ofstream values(file, std::ios::binary);
    if(!values.is_open())
        throw std::runtime_error("Unable to write file " + std::string(file));

    values.write(reinterpret_cast<char*>(data), size);
    values.close();
}


void StartTimer()
{
  LARGE_INTEGER li;
  if(!QueryPerformanceFrequency(&li))
    printf("QueryPerformanceFrequency failed!\n");

  PCFreq = (double)li.QuadPart/1000.0;

  QueryPerformanceCounter(&li);
  timerStart = li.QuadPart;
}

// time elapsed in ms
double GetTimer()
{
  LARGE_INTEGER li;
  QueryPerformanceCounter(&li);
  return (double)(li.QuadPart-timerStart)/PCFreq;
}

#define SOFTENING 1e-7f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

struct Body {
    float x, y, z, vx, vy, vz;
};

/*
 * Calculate the gravitational impact of all bodies in the system
 * on all others.
 */
template < uint32_t BlockSz >
__global__ void bodyForceKernel(Body *InOut, float dt, int n)
{
    uint32_t thid = threadIdx.x;

//    PRINTZ("%d: ii = %d", thid, ii);
//    auto res = prefixSum< BlockSz >(ii, [](uint32_t& lhs, uint32_t rhs) { lhs += rhs; });
//    PRINTZ("%d: ii = %d; res = %d", thid, ii, res);
//    return;

    __shared__ Body B[1];

  // one thread block runs for one 'i'
   // iterations over 'i' are independent
  for (int i = blockIdx.x; i < n; i += gridDim.x) {

    struct UU { float vx, vy, vz; };
    UU S{};

    if(thid == 0)
        B[0] = InOut[i];
    __syncthreads();

    // one block is responsible for inner loop
    for (int j = thid; j < n; j += BlockSz) {
    //for (int j = 0; j < n; j++) {
      float dx = InOut[j].x - InOut[i].x;
      float dy = InOut[j].y - InOut[i].y;
      float dz = InOut[j].z - InOut[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;
      //float dist = sqrtf(distSqr),
       //     invDist3 = 1.0f / (distSqr * dist);

      S.vx += dx * invDist3;
      S.vy += dy * invDist3;
      S.vz += dz * invDist3;
    }
#if 0
    __shared__ UU UUX[BlockSz];
    UUX[thid] = S;
    __syncthreads();

    if(thid == 0) {
        UU res = {};
        for(int j = 0; j < BlockSz; j++) {
            res.vx += UUX[j].vx;
            res.vy += UUX[j].vy;
            res.vz += UUX[j].vz;
        }

        InOut[i].vx = B[0].vx + dt*res.vx;
        InOut[i].vy = B[0].vy + dt*res.vy;
        InOut[i].vz = B[0].vz + dt*res.vz;
    }

#else

    auto cta = this_thread_block();
    // actually we do not need the full prefix sum but just total sum..
    auto res = prefixSum< BlockSz >(cta, S, [](UU& lhs, const UU& rhs)
            { lhs.vx += rhs.vx, lhs.vy += rhs.vy, lhs.vz += rhs.vz; }
    );

    uint32_t numThids = n % BlockSz;
    if(numThids == 0)
        numThids = BlockSz;
    //__syncthreads();
    // n is divided between several threads: in total 4096 / 128

    // the last thread holds the total sum
    if(thid == numThids-1) {
        InOut[i].vx += dt*res.vx;
        InOut[i].vy += dt*res.vy;
        InOut[i].vz += dt*res.vz;
    }
#endif
    __syncthreads();
  } // for i
}

template < uint32_t BlockSz >
__global__ void accumulateKernel(Body *p, float dt, int n)
{
    int idx = blockIdx.x*BlockSz + threadIdx.x,
        step = BlockSz * gridDim.x;
    for (int i = idx; i < n; i += step) { // integrate position
        p[i].x += p[i].vx*dt;
        p[i].y += p[i].vy*dt;
        p[i].z += p[i].vz*dt;
    }
}


#define CUDA_CHECK { \
    auto res = cudaGetLastError(); \
    if(res != cudaSuccess) \
       fprintf(stderr, "%d: CUDA error: %s\n", __LINE__, cudaGetErrorString(res)); \
    }

// http://en.wikipedia.org/wiki/Kahan_summation_algorithm
__device__ __host__ void bodyForce(Body *p, float dt, int n)
{
  for (int i = 0; i < n; ++i) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}


static void computeCPU(const char *fname, Body *p, int nBodies, int nIters, const float dt)
{
    PRINTZ("Computing reference solution");
    int bytes = nBodies*sizeof(Body);
    read_values_from_file(fname, (float *)p, bytes);

    for (int iter = 0; iter < nIters; iter++) {
    /*
     * You will likely wish to refactor the work being done in `bodyForce`,
     * and potentially the work to integrate the positions.
     */

      bodyForce(p, dt, nBodies); // compute interbody forces

    /*
     * This position integration cannot occur until this round of `bodyForce` has completed.
     * Also, the next round of `bodyForce` cannot begin until the integration is complete.
     */

      for (int i = 0 ; i < nBodies; i++) { // integrate position
        p[i].x += p[i].vx*dt;
        p[i].y += p[i].vy*dt;
        p[i].z += p[i].vz*dt;
      }
    }
}

int nBodySim(const int argc, const char** argv)
{
    int deviceId = -1, numberOfSMs = 0;
    cudaGetDevice(&deviceId);                  // deviceId: now points to the id of the currently active GPU.
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    PRINTZ("deviceID: %d; Num SMs: %d", deviceId, numberOfSMs);

    // The assessment will test against both 2<11 and 2<15.
    // Feel free to pass the command line argument 15 when you generate ./nbody report files
    int nBodies = 2<<11;
    if (argc > 1) nBodies = 2<<atoi(argv[1]);

    // The assessment will pass hidden initialized values to check for correctness.
    // You should not make changes to these files, or else the assessment will not work.
    const char * initialized_values;
    const char * solution_values;

    if (nBodies == 2<<11) {
        initialized_values = "initialized_4096";
        solution_values = "solution_4096";
    } else { // nBodies == 2<<15
        initialized_values = "initialized_65536";
        solution_values = "solution_65536";
    }

    if (argc > 2) initialized_values = argv[2];
    if (argc > 3) solution_values = argv[3];

    const float dt = 0.01f; // Time step
    const int nIters = 10;  // Simulation iterations

    Body *p = nullptr;
    auto bytes = nBodies * sizeof(Body);
    cudaMallocManaged(&p, bytes);
    //cudaMemPrefetchAsync(p, bytes, cudaCpuDeviceId);

    read_values_from_file(initialized_values, (float *)p, bytes);

    double totalTime = 0.0;

    /*
   *     This simulation will run for 10 cycles of time, calculating gravitational
    * interaction amongst bodies, and adjusting their positions to reflect.
    */

//  auto err = cudaGetLastError();
//  PRINTZ("last error: %s", cudaGetErrorString(err));

    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();
        //cudaMemPrefetchAsync(p, bytes, deviceId);

        /*
        * You will likely wish to refactor the work being done in `bodyForce`,
        * and potentially the work to integrate the positions.
        */

        constexpr uint32_t BlockSz = 128;
        uint32_t nBlocks = nBodies / BlockSz;
        bodyForceKernel< BlockSz ><<<nBodies, BlockSz>>>(p, dt, nBodies);

        accumulateKernel< BlockSz ><<<nBlocks, BlockSz>>>(p, dt, nBodies);

        cudaDeviceSynchronize();

        //cudaMemPrefetchAsync(p, bytes, cudaCpuDeviceId);

        /*
        * This position integration cannot occur until this round of `bodyForce` has completed.
        * Also, the next round of `bodyForce` cannot begin until the integration is complete.
        */


        CUDA_CHECK
        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }

//  write_values_to_file(solution_values, (float *)p, bytes);
//  return 1;

  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

  std::vector< Body > truth(nBodies);
   //read_values_from_file(solution_values, (float *)p, bytes);
  computeCPU(initialized_values, truth.data(), nBodies, nIters, dt);

  for(uint32_t i = 0; i < truth.size(); i++) {

      const float eps = 1e-2;
      const Body& pT = truth[i], pC = p[i];

      auto F = [&i, &eps](auto check, auto truth) {
         auto d = std::abs(check - truth), abst = std::abs(truth);
         if(abst > 1e-3)
             d /= abst;
         if(d > eps) {
             PRINTZ("%d: %f -- %f: diff: %f", i, truth, check, d);
         }
      };

      F(pC.x, pT.x); F(pC.y, pT.y); F(pC.z, pT.z);
      F(pC.vx, pT.vx); F(pC.vy, pT.vy); F(pC.vz, pT.vz);
  }

  // You will likely enjoy watching this value grow as you accelerate the application,
  // but beware that a failure to correctly synchronize the device might result in
  // unrealistically high values.
  printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

  cudaFree(p);

  return 1;
}
