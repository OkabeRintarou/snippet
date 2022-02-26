#include "common.h"
#include "cpu_anim.h"
#include <cmath>

struct DataBlock {
  unsigned char *dev_bitmap = nullptr;
  CPUAnimBitmap *bitmap = nullptr;
};

void cleanup(void *p) {
  DataBlock &d = *(static_cast<DataBlock *>(p));
  cudaFree(d.dev_bitmap);
}

static constexpr int DIM = 1024;

__global__ void kernel(unsigned char *ptr, int ticks) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  const float fdim = static_cast<float>(DIM);
  float fx = float(x) - fdim / 2.0f;
  float fy = float(y) - fdim / 2.0f;
  float d = sqrtf(fx * fx + fy * fy);
  unsigned char grey = static_cast<unsigned char>(
      128.0f +
      127.0f * cosf(d / 10.0f - float(ticks) / 7.0f) / (d / 10.0f + 1.0f));
  ptr[offset * 4 + 0] = grey;
  ptr[offset * 4 + 1] = grey;
  ptr[offset * 4 + 2] = grey;
  ptr[offset * 4 + 3] = 255;
}

static void generate_frame(void *p, int ticks) {
  auto d = static_cast<DataBlock *>(p);
  dim3 blocks(DIM / 16, DIM / 16);
  dim3 threads(16, 16);

  kernel<<<blocks, threads>>>(d->dev_bitmap, ticks);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(),
                   cudaMemcpyDeviceToHost));
}

int main() {
  DataBlock data;
  CPUAnimBitmap bitmap(DIM, DIM, &data);
  data.bitmap = &bitmap;

  CHECK(cudaMalloc(&data.dev_bitmap, bitmap.image_size()));
  bitmap.anim_and_exit(generate_frame, nullptr);
  return 0;
}
