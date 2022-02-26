#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include "common.h"

static const int N = 256;

__global__ void add(int * dev_a, int *dev_b, int *dev_c) {
  int tid = blockIdx.x;
  if (tid < N) {
    dev_c[tid] = dev_a[tid] + dev_b[tid];
  }
}

int main() {
  cuda::vector<int> dev_a(N), dev_b(N), dev_c(N);
  for (int i = 0; i < N; i++) {
    dev_a[i] = i;
    dev_b[i] = i * i;
  }

  add<<<N, 1>>>(dev_a.data(), dev_b.data(), dev_c.data());

  cudaDeviceSynchronize();

  // check cpu and gpu get the same result
  for (int i = 0; i < N; i++) {
    assert(dev_c[i] == (i + i * i));
  }
  return 0;
}