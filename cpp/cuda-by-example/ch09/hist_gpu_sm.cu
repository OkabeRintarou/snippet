#include <cassert>
#include "common.h"
#include <random>

static constexpr int SIZE = 100 * 1024 * 1024;

__global__ void histo_kernel(unsigned char *buffer, int *histo) {
  __shared__ unsigned int temp[256];
  temp[threadIdx.x] = 0;
  __syncthreads();

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  while (i < SIZE) {
    atomicAdd(&temp[buffer[i]], 1);
    i += stride;
  }
  __syncthreads();
  atomicAdd(&histo[threadIdx.x], temp[threadIdx.x]);
}

int main() {
  cuda::vector<unsigned char> buffer(SIZE);
  cuda::vector<int> histo(256);

  std::fill(std::begin(histo), std::end(histo), 0);
  for (auto &v : buffer) {
    v = rand();
  }
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start, 0));

  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  int blocks = prop.multiProcessorCount;

  histo_kernel<<<blocks * 2, 256>>>(buffer.data(), histo.data());

  CHECK(cudaEventRecord(stop, 0));
  CHECK(cudaEventSynchronize(stop));

  float elapsed_time = .0f;
  CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

  printf("Time to generate: %3.1f ms\n", elapsed_time);

  long histo_count = 0;
  for (int i = 0; i < 256; i++) {
    histo_count += histo[i];
  }
  printf("Histogram Sum: %ld\n", histo_count);

  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  assert(histo_count == SIZE);
  return 0;
}