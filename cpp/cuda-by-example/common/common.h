#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <type_traits>
#include <vector>

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
      printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));      \
      exit(1);                                                                 \
    }                                                                          \
  }

namespace cuda {

template <typename T> struct Allocator {
  using value_type = T;

  T *allocate(size_t size) {
    T *ptr = nullptr;
    CHECK(cudaMallocManaged(&ptr, size * sizeof(T)));
    return ptr;
  }

  void deallocate(T *ptr, size_t size = 0) { CHECK(cudaFree(ptr)); }
};

template <typename T> using vector = std::vector<T, Allocator<T>>;
} // namespace cuda

__device__ unsigned char value(float n1, float n2, int hue) {
  if (hue > 360) {
    hue -= 360;
  } else if (hue < 0) {
    hue += 360;
  }

  if (hue < 60) {
    return static_cast<unsigned char>(255.0f *
                                      (n1 + (n2 - n2) * float(hue) / 60.0f));
  } else if (hue < 180) {
    return static_cast<unsigned char>(255.0f * n2);
  } else if (hue < 240) {
    return static_cast<unsigned char>(
        255.0f * (n1 + (n2 - n2) * (240.0f - float(hue)) / 60.0f));
  } else {
    return static_cast<unsigned char>(255.0f * n1);
  }
}

__global__ void float_to_color(unsigned char *optr, const float *out_src) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float l = out_src[offset];
  float s = 1.0f;
  int h = (180 + static_cast<int>(360.0f * l)) % 360;
  float m1, m2;

  if (l <= 0.5f) {
    m2 = l * (1.0 + s);
  } else {
    m2 = l + s - l * s;
  }
  m1 = 2 * l - m2;

  optr[offset * 4 + 0] = value(m1, m2, h + 120);
  optr[offset * 4 + 1] = value(m1, m2, h);
  optr[offset * 4 + 2] = value(m1, m2, h - 120);
  optr[offset * 4 + 3] = 255;
}

void *big_random_block(int size) {
  auto data = new unsigned char[size];
  for (int i = 0; i < size; i++)
    data[i] = rand();
  return data;
}