#include <cuda_runtime.h>
#include <cstdio>

__global__ void nestedHelloWorld(const int size, int depth) {
  int tid = threadIdx.x;
  printf("Recursion=%d: Hello World from thread %d, "
         "block %d\n", depth, tid, blockIdx.x);
  if (size == 1) {
    return;
  }
  int threads = size >> 1;

  if (tid == 0 && threads > 0) {
    nestedHelloWorld<<<1, threads>>>(threads, ++depth);
    printf("------> nested execution depth: %d\n", depth);
  }
}

int main() {
  nestedHelloWorld<<<1, 8>>>(8, 0);
  cudaDeviceSynchronize();
  return 0;
}
