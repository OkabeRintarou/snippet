#include "common.h"
#include "cpu_bitmap.h"

static const int DIM = 1000;

struct cuComplex {
  float r;
  float i;
  __device__ cuComplex(float a, float b) : r(a), i(b) {}
  __device__ float magnitude2() const { return r * r + i * i; }
  __device__ cuComplex operator*(const cuComplex &a) {
    return {r * a.r - i * a.i, i * a.r + r * a.i};
  }
  __device__ cuComplex operator+(const cuComplex &a) {
    return {r + a.r, i + a.i};
  }
};

__device__ int julia(int x, int y) {
  const float scale = 1.5f;
  const auto dim = static_cast<float>(DIM);
  float jx = scale * (dim / 2.f - float(x)) / (dim / 2.f);
  float jy = scale * (dim / 2.f - float(y)) / (dim / 2.f);

  cuComplex c(-0.8f, 0.156f);
  cuComplex a(jx, jy);

  for (int i = 0; i < 200; i++) {
    a = a * a + c;
    if (a.magnitude2() > 1000.f) {
      return 0;
    }
  }
  return 1;
}

__global__ void kernel(unsigned char *ptr) {
  int x = blockIdx.x;
  int y = blockIdx.y;

  int offset = x + y * gridDim.x;
  int julia_value = julia(x, y);
  ptr[offset * 4 + 0] = 255 * julia_value;
  ptr[offset * 4 + 1] = 0;
  ptr[offset * 4 + 2] = 0;
  ptr[offset * 4 + 3] = 255;
}

int main() {
  CPUBitmap bitmap(DIM, DIM);
  unsigned char *dev_bitmap;

  CHECK(cudaMalloc(&dev_bitmap, bitmap.image_size()));

  dim3 grid(DIM, DIM);
  kernel<<<grid, 1>>>(dev_bitmap);
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(),
                   cudaMemcpyDeviceToHost));
  bitmap.display_and_exit();
  cudaFree(dev_bitmap);
  return 0;
}