#include "common.h"

static const int SIZE = (64 * 1024 * 1024);

float cuda_malloc_test(int size, bool up) {
  cudaEvent_t start, stop;
  int *a, *dev_a;
  float elapsed_time;
  const int bytes = size * sizeof(int);

  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  a = new int[size];
  CHECK(cudaMalloc(&dev_a, bytes));

  cudaEventRecord(start, 0);
  for (int i = 0; i < 100; i++) {
    if (up) {
      CHECK(cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice));
    } else {
      CHECK(cudaMemcpy(a, dev_a, bytes, cudaMemcpyDeviceToHost));
    }
  }

  CHECK(cudaEventRecord(stop, 0));
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

  delete[] a;
  CHECK(cudaFree(dev_a));
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));

  return elapsed_time;
}

float cuda_host_malloc_test(int size, bool up) {
  cudaEvent_t start, stop;
  int *a, *dev_a;
  float elapsed_time;
  const int bytes = size * sizeof(int);

  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaHostAlloc(&a, bytes, cudaHostAllocDefault));
  CHECK(cudaMalloc(&dev_a, bytes));

  cudaEventRecord(start, 0);
  for (int i = 0; i < 100; i++) {
    if (up) {
      CHECK(cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice));
    } else {
      CHECK(cudaMemcpy(a, dev_a, bytes, cudaMemcpyDeviceToHost));
    }
  }

  CHECK(cudaEventRecord(stop, 0));
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

  CHECK(cudaFreeHost(a));
  CHECK(cudaFree(dev_a));
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));

  return elapsed_time;
}

int main() {
  float elapsed_time;
  float MB = float(100) * SIZE * sizeof(int) / 1024.0f / 1024.0f;

  elapsed_time = cuda_malloc_test(SIZE, true);
  printf("Time using cudaMalloc:  %3.1f ms\n", elapsed_time);
  printf("\t MB/s during copy up:  %3.1f\n", MB / (elapsed_time / 1000.f));

  elapsed_time = cuda_malloc_test(SIZE, false);
  printf("Time using cudaMalloc:  %3.1f ms\n", elapsed_time);
  printf("\t MB/s during copy down:  %3.1f\n", MB / (elapsed_time / 1000.f));

  elapsed_time = cuda_host_malloc_test(SIZE, true);
  printf("Time using cudaHostMalloc:  %3.1f ms\n", elapsed_time);
  printf("\t MB/s during copy up:  %3.1f\n", MB / (elapsed_time / 1000.f));

  elapsed_time = cuda_host_malloc_test(SIZE, true);
  printf("Time using cudaHostMalloc:  %3.1f ms\n", elapsed_time);
  printf("\t MB/s during copy down:  %3.1f\n", MB / (elapsed_time / 1000.f));
  return 0;
}
