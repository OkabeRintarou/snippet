#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main() {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess) {
    fprintf(stderr, "cudaGetDeviceCount returned %d\n-> %s\n",
            (int)error,
            cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }

  if (device_count == 0) {
    fprintf(stderr, "There are no available device(s) that support CUDA\n");
  } else {
    printf("Detected %d CUDA Capable device(s)\n", device_count);
  }

  int dev = 0, driver_version = 0, runtime_version = 0;

  cudaSetDevice(dev);
  cudaDeviceProp device_prop{};
  cudaGetDeviceProperties(&device_prop, dev);
  printf("Device %d: \"%s\"", dev, device_prop.name);

  cudaDriverGetVersion(&driver_version);
  cudaRuntimeGetVersion(&runtime_version);
  printf("\tCUDA Driver Version / Runtime Version\t\t%d.%d / %d.%d\n",
         driver_version / 1000, (driver_version % 100) / 10,
         runtime_version / 1000, (runtime_version % 100) / 10);
  printf("\tCUDA Capability Major/Minor version number:\t\t%d.%d\n",
         device_prop.major, device_prop.minor);
  printf("\tTotal amount of global memory:\t\t%.2f MBytes (%llu bytes)\n",
         (float)device_prop.totalGlobalMem / (pow(1024.0, 3)),
         (unsigned long long)device_prop.totalGlobalMem);
  printf("\tGPU Clock rate:\t\t%.0f MHz (%0.2f GHz)\n",
         (float)device_prop.clockRate * 1e-3f,
         (float)device_prop.clockRate * 1e-6f);
  printf("\tMemory Clock rate:\t\t%.0f MHz\n",
         (float)device_prop.memoryClockRate * 1e-3f);
  printf("\tMemory Bus Width:\t\t %d-bit\n", device_prop.memoryBusWidth);
  if (device_prop.l2CacheSize) {
    printf("\tL2 Cache Size:\t\t%d bytes\n",
           device_prop.l2CacheSize);
  }
  printf("\tMax Texture Dimension Size (x, y, z)\t\t"
         "1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
         device_prop.maxTexture1D,
         device_prop.maxTexture2D[0], device_prop.maxTexture2D[1],
         device_prop.maxTexture3D[0], device_prop.maxTexture3D[1], device_prop.maxTexture3D[2]);
  printf("\tMax Layered Texture Size (dim) x layers\t\t"
         "1D=(%d) x %d, 2D=(%d,%d) x %d\n",
         device_prop.maxTexture1DLayered[0],
         device_prop.maxTexture1DLayered[1],
         device_prop.maxTexture2DLayered[0],
         device_prop.maxTexture2DLayered[1],
         device_prop.maxTexture2DLayered[2]);
  printf("\tTotal amount of constant memory:\t\t%lu bytes\n", device_prop.totalConstMem);
  printf("\tTotal amount of shared memory:\t\t%lu bytes\n", device_prop.sharedMemPerBlock);
  printf("\tTotal number of registers available per multiprocessor:\t\t%d\n", device_prop.regsPerMultiprocessor);
  printf("\tTotal number of registers available per block:\t\t%d\n", device_prop.regsPerBlock);
  printf("\tWarp size:\t\t%d\n", device_prop.warpSize);
  printf("\tMaximum number of threads per block:\t\t%d\n", device_prop.maxThreadsPerBlock);
  printf("\tMaximum number of threads per multiprocessor:\t\t%d\n", device_prop.maxThreadsPerMultiProcessor);
  printf("\tMaximum sizes of each dimension of a block:\t\t%d x %d x %d\n",
         device_prop.maxThreadsDim[0],
         device_prop.maxThreadsDim[1],
         device_prop.maxThreadsDim[2]);
  printf("\tMaximum sizes of each dimension of a grid:\t\t %d x %d x %d\n",
         device_prop.maxGridSize[0],
         device_prop.maxGridSize[1],
         device_prop.maxGridSize[2]);
  printf("\tMaximum memory pitch:\t\t%d\n", device_prop.memPitch);
  return 0;
}
