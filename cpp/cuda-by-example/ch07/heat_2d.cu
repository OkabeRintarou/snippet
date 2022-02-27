#include "common.h"
#include "cpu_anim.h"
#include <vector>

static constexpr int DIM = 1024;
static constexpr float MAX_TEMP = 1.0f;
static constexpr float MIN_TEMP = 0.0001f;
static constexpr float SPEED = 0.25f;

texture<float, 2> tex_const;
texture<float, 2> tex_in;
texture<float, 2> tex_out;

__global__ void copy_const_kernel(float *iptr) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float c = tex2D(tex_const, x, y);
  if (c != 0.0f) {
    iptr[offset] = c;
  }
}

__global__ void blend_kernel(float *out, bool dst_out) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float t, l, c, r, b;
  if (dst_out) {
    t = tex2D(tex_in, x, y - 1);
    l = tex2D(tex_in, x - 1, y);
    c = tex2D(tex_in, x, y);
    r = tex2D(tex_in, x + 1, y);
    b = tex2D(tex_in, x, y + 1);
  } else {
    t = tex2D(tex_out, x, y - 1);
    l = tex2D(tex_out, x - 1, y);
    c = tex2D(tex_out, x, y);
    r = tex2D(tex_out, x + 1, y);
    b = tex2D(tex_out, x, y + 1);
  }
  out[offset] = c + SPEED * (t + b + r + l - 4.0f * c);
}

struct DataBlock {
  unsigned char *output_bitmap;
  float *dev_in;
  float *dev_out;
  float *dev_const;

  CPUAnimBitmap *bitmap;

  cudaEvent_t start, stop;
  float total_time = .0f;
  float frames = .0f;

  void init(CPUAnimBitmap *bm) {
    bitmap = bm;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    const long image_size = bitmap->image_size();
    CHECK(cudaMalloc(&output_bitmap, image_size));
    CHECK(cudaMalloc(&dev_in, image_size));
    CHECK(cudaMalloc(&dev_out, image_size));
    CHECK(cudaMalloc(&dev_const, image_size));

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    CHECK(cudaBindTexture2D(nullptr, tex_const, dev_const, desc, DIM, DIM,
                            sizeof(float) * DIM));
    CHECK(cudaBindTexture2D(nullptr, tex_in, dev_in, desc, DIM, DIM,
                            sizeof(float) * DIM));
    CHECK(cudaBindTexture2D(nullptr, tex_out, dev_out, desc, DIM, DIM,
                            sizeof(float) * DIM));

    std::vector<float> temp(DIM * DIM);
    for (int i = 0; i < DIM * DIM; i++) {
      temp[i] = 0;
      int x = i % DIM;
      int y = i / DIM;
      if ((x > 300) && (x < 600) && (y > 310) && (y < 601)) {
        temp[i] = MAX_TEMP;
      }
    }

    temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2.0f;
    temp[DIM * 700 + 100] = MIN_TEMP;
    temp[DIM * 300 + 300] = MIN_TEMP;
    temp[DIM * 200 + 700] = MIN_TEMP;
    for (int y = 800; y < 900; y++) {
      for (int x = 400; x < 500; x++) {
        temp[x + y * DIM] = MIN_TEMP;
      }
    }

    CHECK(cudaMemcpy(dev_const, temp.data(), bitmap->image_size(),
                     cudaMemcpyHostToDevice));
    for (int y = 800; y < DIM; y++) {
      for (int x = 0; x < 200; x++) {
        temp[x + y * DIM] = MAX_TEMP;
      }
    }
    CHECK(cudaMemcpy(dev_in, temp.data(), bitmap->image_size(),
                     cudaMemcpyHostToDevice));
  }

  ~DataBlock() {
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(dev_in));
    CHECK(cudaFree(dev_out));
    CHECK(cudaFree(dev_const));
  }
};

void anim_gpu(void *p, int) {
  auto d = static_cast<DataBlock *>(p);
  CHECK(cudaEventRecord(d->start, 0));
  dim3 blocks(DIM / 16, DIM / 16);
  dim3 threads(16, 16);
  CPUAnimBitmap *bitmap = d->bitmap;

  volatile bool dst_out = true;
  for (int i = 0; i < 90; i++) {
    float *in, *out;
    if (dst_out) {
      in = d->dev_in;
      out = d->dev_out;
    } else {
      in = d->dev_out;
      out = d->dev_in;
    }
    copy_const_kernel<<<blocks, threads>>>(in);
    blend_kernel<<<blocks, threads>>>(out, dst_out);
    dst_out = !dst_out;
  }
  float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_in);
  CHECK(cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaEventRecord(d->stop, 0));
  CHECK(cudaEventSynchronize(d->stop));
  float elapsed_time;
  CHECK(cudaEventElapsedTime(&elapsed_time, d->start, d->stop));

  d->total_time += elapsed_time;
  ++d->frames;
  printf("Average Time per frame: %3.1f ms\n", d->total_time / d->frames);
}

void anim_exit(void *) {
  CHECK(cudaUnbindTexture(tex_in));
  CHECK(cudaUnbindTexture(tex_out));
  CHECK(cudaUnbindTexture(tex_const));
}

int main() {
  DataBlock data;
  CPUAnimBitmap bitmap(DIM, DIM, &data);
  data.init(&bitmap);
  bitmap.anim_and_exit(anim_gpu, anim_exit);
  return 0;
}
