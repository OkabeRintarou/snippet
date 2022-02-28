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

  cudaStream_t stream0, stream1;
  CHECK(cudaStreamCreate(&stream0));
  CHECK(cudaStreamCreate(&stream1));

  int *host_a, *host_b, *host_c;
  int *dev_a0, *dev_b0, *dev_c0;
  int *dev_a1, *dev_b1, *dev_c1;

  CHECK(cudaMalloc(&dev_a0, N * sizeof(int)));
  CHECK(cudaMalloc(&dev_b0, N * sizeof(int)));
  CHECK(cudaMalloc(&dev_c0, N * sizeof(int)));
  CHECK(cudaMalloc(&dev_a1, N * sizeof(int)));
  CHECK(cudaMalloc(&dev_b1, N * sizeof(int)));
  CHECK(cudaMalloc(&dev_c1, N * sizeof(int)));

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
  for (int i = 0; i < FULL_DATA_SIZE; i += N * 2) {
    CHECK(cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int),
                          cudaMemcpyHostToDevice, stream0));
    CHECK(cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int),
                          cudaMemcpyHostToDevice, stream0));

    CHECK(cudaMemcpyAsync(dev_a1, host_a + i + N, N * sizeof(int),
                          cudaMemcpyHostToDevice, stream1));
    CHECK(cudaMemcpyAsync(dev_b1, host_b + i + N, N * sizeof(int),
                          cudaMemcpyHostToDevice, stream1));


    kernel<<<N / 256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_c0);
    kernel<<<N / 256, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);

    CHECK(cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int),
                          cudaMemcpyDeviceToHost, stream0));
    CHECK(cudaMemcpyAsync(host_c + i + N, dev_c1, N * sizeof(int),
                          cudaMemcpyDeviceToHost, stream1));
  }

  CHECK(cudaStreamSynchronize(stream0));
  CHECK(cudaStreamSynchronize(stream1));
  CHECK(cudaEventRecord(stop, 0));
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

  printf("Time taken:  %3.1f ms\n", elapsed_time);

  CHECK(cudaFreeHost(host_a));
  CHECK(cudaFreeHost(host_b));
  CHECK(cudaFreeHost(host_c));
  CHECK(cudaFree(dev_a0));
  CHECK(cudaFree(dev_b0));
  CHECK(cudaFree(dev_c0));
  CHECK(cudaFree(dev_a1));
  CHECK(cudaFree(dev_b1));
  CHECK(cudaFree(dev_c1));
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  CHECK(cudaStreamDestroy(stream0));
  CHECK(cudaStreamDestroy(stream1));
  return 0;
}
