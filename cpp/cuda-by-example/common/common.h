#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
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

template<typename T>
struct Allocator {
  using value_type = T;

  T *allocate(size_t size) {
    T *ptr = nullptr;
    CHECK(cudaMallocManaged(&ptr, size * sizeof(T)));
    return ptr;
  }

  void deallocate(T *ptr, size_t size = 0) {
    CHECK(cudaFree(ptr));
  }
};

template<typename T>
using vector = std::vector<T, Allocator<T>>;
}
