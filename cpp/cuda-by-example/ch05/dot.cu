#include "common.h"
#include <cassert>

static constexpr int N = 33 * 1024;
static constexpr int threadsPerBlock = 256;
static constexpr int blocksPerGrid =
    std::min(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float *a, float *b, float *c) {
  __shared__ float cache[threadsPerBlock];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;
  float temp = .0f;
  while (tid < N) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }
  cache[cacheIndex] = temp;

  __syncthreads();

  // reductions, threadsPerBlock must be a power of 2
  int i = blockDim.x / 2;
  while (i != 0) {
    if (cacheIndex < i) {
      cache[cacheIndex] += cache[cacheIndex + i];
    }
    __syncthreads();
    i /= 2;
  }

  if (cacheIndex == 0) {
    c[blockIdx.x] = cache[0];
  }
}

static inline float sum_squares(unsigned long long x) {
  float sum = 0.0f;
  for (int i = 0; i < N; i++) {
    float dot = static_cast<float>(i) * static_cast<float>(i * 2);
    sum += dot;
  }
  return sum;
}

int main() {
  cuda::vector<float> a(N), b(N), partial_c(blocksPerGrid);

  for (int i = 0; i < N; i++) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i * 2);
  }

  dot<<<blocksPerGrid, threadsPerBlock>>>(a.data(), b.data(), partial_c.data());

  CHECK(cudaDeviceSynchronize());

  float gpu_sum = .0f;
  for (int i = 0; i < blocksPerGrid; i++) {
    gpu_sum += partial_c[i];
  }

  float cpu_sum = sum_squares(N);
  printf("gpu_sum = %.6g, cpu_sum = %.6g\n", gpu_sum, cpu_sum);
  return 0;
}
