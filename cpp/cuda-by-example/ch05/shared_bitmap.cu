#include "common.h"
#include "cpu_bitmap.h"

static const int DIM = 1024;
static const float PI = 3.1415926535897932f;

__global__ void kernel(unsigned char *ptr) {
  __shared__ float shared[16][16];

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;
  const float period = 128.0f;
  const float fx = static_cast<float>(x);
  const float fy = static_cast<float>(y);

  shared[threadIdx.x][threadIdx.y] =
      255.0f * (sinf(fx * 2.0f * PI / period) + 1.0f) *
      (sinf(fy * 2.0f * PI / period) + 1.0f) / 4.0f;

  // add a synchronization point between the write to
  // shared memory and subsequent read from it
  __syncthreads();

  ptr[offset * 4 + 0] = 0;
  ptr[offset * 4 + 1] = shared[15 - threadIdx.x][15 - threadIdx.y];
  ptr[offset * 4 + 2] = 0;
  ptr[offset * 4 + 3] = 255;
}

int main() {
  CPUBitmap bitmap(DIM, DIM);
  unsigned char *dev_bitmap;

  CHECK(cudaMalloc(&dev_bitmap, bitmap.image_size()));

  dim3 grids(DIM / 16, DIM / 16);
  dim3 threads(16, 16);
  kernel<<<grids, threads>>>(dev_bitmap);
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(),
                   cudaMemcpyDeviceToHost));
  bitmap.display_and_exit();
  cudaFree(dev_bitmap);
  return 0;
}
