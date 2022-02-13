#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
      printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));      \
      exit(1);                                                                 \
    }                                                                          \
  }
void printDeviceProperties(const cudaDeviceProp &prop, int index) {
  printf("    ---- General Information for device %d ----\n", index);
  printf("Name: %s\n", prop.name);
  printf("Compute capability:\t%d.%d\n", prop.major, prop.minor);
  printf("Clock rate: %d\n", prop.clockRate);
  printf("Device copy overlap:   ");
  if (prop.deviceOverlap) {
    printf("Enabled\n");
  } else {
    printf("Disabled\n");
  }
  printf("Kernel execition timeout : ");
  if (prop.kernelExecTimeoutEnabled) {
    printf("Enabled\n");
  } else {
    printf("Disabled\n");
  }
  printf("    ---- Memory Information for device %d ----\n", index);
  printf("Total global mem: %ld\n", prop.totalGlobalMem);
  printf("Total constant mem: %ld\n", prop.totalConstMem);
  printf("Max mem pitch: %ld\n", prop.memPitch);
  printf("Texture Alignment: %ld\n", prop.textureAlignment);

  printf("    ---- MP Information for device %d ----\n", index);
  printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
  printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
  printf("Register per mp: %d\n", prop.regsPerBlock);
  printf("Threads in warp: %d\n", prop.warpSize);
  printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
  printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0],
         prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0],
         prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("\n");
}

int main() {
  cudaDeviceProp prop;
  int count = 0;

  cudaGetDeviceCount(&count);
  for (int i = 0; i < count; i++) {
    CHECK(cudaGetDeviceProperties(&prop, i));
    printDeviceProperties(prop, i);
  }
  return 0;
}
