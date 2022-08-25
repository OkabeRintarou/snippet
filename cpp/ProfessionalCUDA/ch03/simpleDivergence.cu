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

__global__ void mathKernel3(float *c) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float a = 0.0f, b = 0.0f;

  bool ipred = (tid % 2) == 0;

  if (ipred) {
    a = 100.0f;
  }
  if (!ipred) {
    b = 200.0f;
  }
  c[tid] = a + b;
}
__global__ void mathKernel4(float *c) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float a = 0.0f, b = 0.0f;

  int itid = tid >> 5;

  if (itid & 0x01 == 0) {
    a = 100.0f;
  } else {
    b = 200.0f;
  }

  c[tid] = a + b;
}

__global__ void warmingup(float *c) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float ia = 0.0f, ib = 0.0f;

  if ((tid / warpSize) % 2 == 0) {
    ia = 100.0f;
  } else {
    ib = 200.0f;
  }
  c[tid] = ia + ib;
}

int main(int argc, char *argv[]) {

  int size = 10240000;
  int blockSize = 64;

  const int dev = 0;
  cudaDeviceProp device_prop{};
  cudaGetDeviceProperties(&device_prop, dev);
  printf("%s using Device %d: %s\n", argv[0], dev, device_prop.name);

  if (argc > 1) {
    blockSize = atoi(argv[1]);
  }
  if (argc > 2) {
    size = atoi(argv[2]);
  }

  dim3 block(blockSize, 1);
  dim3 grid((size + block.x - 1) / block.x, 1);
  printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

  using namespace thrust;
  {
    Timer timer;
    device_vector<float> device(size);
    float *device_ptr = raw_pointer_cast(device.data());
    timer.reset();
    warmingup<<<grid, block>>>(device_ptr);
    cudaDeviceSynchronize();
    printf("warmup<<<%4d %4d>>> elapsed %lf ns\n",
           grid.x, block.x, timer.elapsed_nanoseconds());
  }

  {
    Timer timer;
    device_vector<float> device(size);
    float *device_ptr = raw_pointer_cast(device.data());
    timer.reset();
    mathKernel1<<<grid, block>>>(device_ptr);
    cudaDeviceSynchronize();
    printf("mathKernel1<<<%4d %4d>>> elapsed %lf ns\n",
           grid.x, block.x, timer.elapsed_nanoseconds());
  }

  {
    Timer timer;
    device_vector<float> device(size);
    float *device_ptr = raw_pointer_cast(device.data());
    timer.reset();
    mathKernel2<<<grid, block>>>(device_ptr);
    cudaDeviceSynchronize();
    printf("mathKernel2<<<%4d %4d>>> elapsed %lf ns\n",
           grid.x, block.x, timer.elapsed_nanoseconds());
  }

  {
    Timer timer;
    device_vector<float> device(size);
    float *device_ptr = raw_pointer_cast(device.data());
    timer.reset();
    mathKernel3<<<grid, block>>>(device_ptr);
    cudaDeviceSynchronize();
    printf("mathKernel3<<<%4d %4d>>> elapsed %lf ns\n",
           grid.x, block.x, timer.elapsed_nanoseconds());
  }
  {
    Timer timer;
    device_vector<float> device(size);
    float *device_ptr = raw_pointer_cast(device.data());
    timer.reset();
    mathKernel4<<<grid, block>>>(device_ptr);
    cudaDeviceSynchronize();
    printf("mathKernel4<<<%4d %4d>>> elapsed %lf ns\n",
           grid.x, block.x, timer.elapsed_nanoseconds());
  }
  return 0;
}
