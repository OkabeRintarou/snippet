#include "common.h"
#include <cassert>

static const int N = 8488481;

__global__ void kernel(int *a, int *b, int *c) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < N) {
    c[tid] = a[tid] + b[tid];
    tid += blockDim.x * gridDim.x;
  }
}

int main() {
  cuda::vector<int> a(N), b(N), c(N);
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i * i;
  }
  kernel<<<(N + 127) / 128, 128>>>(a.data(), b.data(), c.data());
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorName(err));
  }
  CHECK(cudaDeviceSynchronize());
  for (int i = 0; i < N; i++) {
    assert(c[i] == i * i + i);
  }
  return 0;
}