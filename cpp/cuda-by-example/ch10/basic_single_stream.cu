#include "common.h"

static constexpr int N = 1024 * 1024;
static constexpr int FULL_DATA_SIZE = N * 20;

__global__ void kernel(int *a, int *b, int *c) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    int idx1 = (idx + 1) % 256;
    int idx2 = (idx + 2) % 256;
    float as = (float)(a[idx] + a[idx1] + a[idx2]) / 3.0f;
    float bs = (float)(b[idx] + b[idx1] + b[idx2]) / 3.0f;
    c[idx] = int((as + bs) / 2.0f);
  }
}

int main() {
  cudaDeviceProp prop;
  int which_device;
  CHECK(cudaGetDevice(&which_device));
  CHECK(cudaGetDeviceProperties(&prop, which_device));
  if (!prop.deviceOverlap) {
    printf("Device will not handle overlaps, so no "
           "speed up from streams\n");
    return 0;
  }
  cudaEvent_t start, stop;
  float elapsed_time;

  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  int *host_a, *host_b, *host_c;
  int *dev_a, *dev_b, *dev_c;

  CHECK(cudaMalloc(&dev_a, N * sizeof(int)));
  CHECK(cudaMalloc(&dev_b, N * sizeof(int)));
  CHECK(cudaMalloc(&dev_c, N * sizeof(int)));

  CHECK(cudaHostAlloc(&host_a, FULL_DATA_SIZE * sizeof(int),
                      cudaHostAllocDefault));
  CHECK(cudaHostAlloc(&host_b, FULL_DATA_SIZE * sizeof(int),
                      cudaHostAllocDefault));
  CHECK(cudaHostAlloc(&host_c, FULL_DATA_SIZE * sizeof(int),
                      cudaHostAllocDefault));

  for (int i = 0; i < FULL_DATA_SIZE; i++) {
    host_a[i] = rand();
    host_b[i] = rand();
  }

  CHECK(cudaEventRecord(start, 0));
  for (int i = 0; i < FULL_DATA_SIZE; i += N) {
    CHECK(cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int),
                          cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int),
                          cudaMemcpyHostToDevice, stream));

    kernel<<<N / 256, 256, 0, stream>>>(dev_a, dev_b, dev_c);

    CHECK(cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int),
                          cudaMemcpyDeviceToHost, stream));
  }

  CHECK(cudaStreamSynchronize(stream));
  CHECK(cudaEventRecord(stop, 0));
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

  printf("Time taken:  %3.1f ms\n", elapsed_time);

  CHECK(cudaFreeHost(host_a));
  CHECK(cudaFreeHost(host_b));
  CHECK(cudaFreeHost(host_c));
  CHECK(cudaFree(dev_a));
  CHECK(cudaFree(dev_b));
  CHECK(cudaFree(dev_c));
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  CHECK(cudaStreamDestroy(stream));
  return 0;
}
