#include <cstdio>
#include <cuda_runtime.h>

__global__ void add(int a, int b, int *sum) {
  *sum = a + b;
}

int main() {
  int *result;
  cudaMalloc((void**)&result, sizeof(int));
  add<<<1, 1>>>(100, 200, result);
  cudaDeviceSynchronize();

  int h_result = 0;
  cudaMemcpy(&h_result, result, sizeof(int), cudaMemcpyDeviceToHost);
  printf("100 + 200 = %d\n", h_result);
  return 0;
}