#include "hip/hip_runtime.h"
#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
#include "../gpu-metrics/gpu-metrics.hpp"
#include <iomanip>
#include <iostream>

using namespace std;

#ifdef __NVCC__
using dtype = double;
#else
using dtype = float4;
#endif
dtype *dA, *dB;

__global__ void initKernel(dtype *A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = dtype(1.1);
  }
}

template <int N>
__global__ void sumKernel(dtype *__restrict__ A, const dtype *__restrict__ B,
                          int blockRun, int blockSize) {
  dtype localSum = dtype(0);

  for (int i = 0; i < N; i++) {
    int idx = blockDim.x * blockRun * i + (blockIdx.x % blockRun) * blockSize +
              threadIdx.x;
    localSum += B[idx];
    // A[idx] = dtype(1.23) * B[idx];
  }
  localSum *= (dtype)1.3;
  if (threadIdx.x > 1233 || localSum == (dtype)23.12)
    A[threadIdx.x] += localSum;
}

template <int N>
double callKernel(int blockCount, int blockRun, int blockSize) {
  sumKernel<N><<<blockCount, blockSize>>>(dA, dB, blockRun, blockSize);
  GPU_ERROR(hipPeekAtLastError());
  return 0.0;
}

template <int N> 
void measure(int blockRun, int blockCount, int blockSize) {

  //int maxActiveBlocks = 0;
  //GPU_ERROR(hipOccupancyMaxActiveBlocksPerMultiprocessor(
  //    &maxActiveBlocks, sumKernel<N, blockSize>, blockSize, 0));

  // GPU_ERROR(hipDeviceSetCacheConfig(hipFuncCachePreferShared));

  MeasurementSeries time;

  GPU_ERROR(hipDeviceSynchronize());
  for (int i = 0; i < 7; i++) {
    const size_t bufferCount = blockRun * blockSize * N + i * 128;
    GPU_ERROR(hipMalloc(&dA, bufferCount * sizeof(dtype)));
    initKernel<<<52, 256>>>(dA, bufferCount);
    GPU_ERROR(hipMalloc(&dB, bufferCount * sizeof(dtype)));
    initKernel<<<52, 256>>>(dB, bufferCount);
    GPU_ERROR(hipDeviceSynchronize());

    double t1 = dtime();
    callKernel<N>(blockCount, blockRun, blockSize);
    GPU_ERROR(hipDeviceSynchronize());
    double t2 = dtime();
    time.add(t2 - t1);

    GPU_ERROR(hipFree(dA));
    GPU_ERROR(hipFree(dB));
  }

  double blockDV = N * blockSize * sizeof(dtype);

  double bw = blockDV * blockCount / time.minValue() / 1.0e9;
  double blockRunValue = blockDV * blockRun / 1024;

  if(blockRunValue > 400000 && blockRunValue < 450000) {
    cout<< setw(10) << bw << "\n"                                  //
        << fixed << setprecision(0) << setw(6);
  }
}

size_t constexpr expSeries(size_t N) {
  size_t val = 20;
  for (size_t i = 0; i < N; i++) {
    val = val * 1.04 + 1;
  }
  return val;
}

int main(int argc, char **argv) {
  initMeasureMetric();

  // Check if blockCount and blockSize arguments are provided
  if(argc < 3) {
    cout << "Please provide blockCount and blockSize as command-line arguments." << endl;
    return 1;
  }

  hipDeviceProp_t prop;
  int deviceId;
  GPU_ERROR(hipGetDevice(&deviceId));
  GPU_ERROR(hipGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;

  // Convert blockSize argument to integer
  int blockSize = 64*atoi(argv[2]);

  for (int blockCount = smCount; blockCount <= smCount*atoi(argv[1]); blockCount += smCount) {
    for (int i = 2; i < 20000; i += max(1.0, i * 0.1)) {
      measure<24>(i, blockCount, blockSize);
    }
  }
}