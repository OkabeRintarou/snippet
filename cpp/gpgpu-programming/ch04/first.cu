#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void my_first_kernel(float *x) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  x[tid] = float(threadIdx.x);
}

int main() {
  float *h_ptr, *d_ptr;
  int blocks, threads, size, n;

  blocks = 2;
  threads = 8;
  size = blocks * threads;

  h_ptr = (float*)malloc(sizeof(float) * size);
  cudaMalloc((void**)&d_ptr, size * sizeof(float));

  my_first_kernel<<<blocks, threads>>>(d_ptr);
  cudaDeviceSynchronize();
  cudaMemcpy(h_ptr, d_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);

  for (n = 0; n < size; n++) {
    printf("%d %f\n", n, h_ptr[n]);
  }

  free(h_ptr);
  cudaFree(d_ptr);
  return 0;
}
