#include "timer.h"
#include <cstdio>
#include <thrust/device_vector.h>

__global__ void mathKernel1(float *c) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float a = 0.0f, b = 0.0f;

  if (tid % 2 == 0) {
    a = 100.0f;
  } else {
    b = 200.0f;
  }
  c[tid] = a + b;
}

__global__ void mathKernel2(float *c) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float a = 0.0f, b = 0.0f;

  if ((tid / warpSize) % 2 == 0) {
    a = 100.0f;
  } else {
    b = 200.0f;
  }
  c[tid] = a + b;
}

int main() {
  int size = 10240000;
  int blockSize = 64;

  dim3 block(blockSize, 1);
  dim3 grid((size + block.x - 1) / block.x, 1);
  printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

  thrust::device_vector<float> d_c1(size), d_c2(size);
  float *d_c1_ptr = thrust::raw_pointer_cast(d_c1.data());
  float *d_c2_ptr = thrust::raw_pointer_cast(d_c2.data());
  Timer timer;

  // run kernel1
  timer.reset();
  mathKernel1<<<grid, block>>>(d_c1_ptr);
  cudaDeviceSynchronize();
  double elapsed1 = timer.elapsed_nanoseconds();
  printf("mathKernel1 <<< %4d, %4d >>> elapsed %lf ns\n", grid.x, block.x,
         elapsed1);

  // run kernel2
  timer.reset();
  mathKernel2<<<grid, block>>>(d_c2_ptr);
  cudaDeviceSynchronize();
  double elapsed2 = timer.elapsed_nanoseconds();
  printf("mathKernel2 <<< %4d, %4d >>> elapsed %lf ns\n", grid.x, block.x,
         elapsed2);

  thrust::host_vector<float> h_c1 = d_c1, h_c2 = d_c2;
  for (size_t i = 0, e = h_c1.size(); i < e; i++) {
    if (h_c1[i] != h_c2[i]) {
      printf("%d %.2f %.2f\n", i, h_c1[i], h_c2[i]);
    }
  }
  return 0;
}
