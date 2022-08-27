#include "common.h"
#include "timer.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <vector>

void initialData(float *f, const int size) {
  for (int i = 0; i < size; i++) {
    f[i] = static_cast<float>(rand() & 0xff) / 10.0f;
  }
}

// clang-format off
void sumMatrixOnHost(const float *a, const float *b, float *c,
                     const int nx, const int ny) {
  // clang-format on
  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      c[ix] = a[ix] + b[ix];
    }
    a += nx;
    b += nx;
    c += nx;
  }
}

void checkResult(float *host_ref, float *gpu_ref, const int N) {
  double epsilon = 1.0E-8;

  for (int i = 0; i < N; i++) {
    if (abs(host_ref[i] - gpu_ref[i]) > epsilon) {
      printf("host %f gpu %f Arrays do not match.\n\n", host_ref[i],
             gpu_ref[i]);
      break;
    }
  }
}

__global__ void sumMatrixOnGPU2D(const float *a, const float *b, float *c,
                                 const int nx, const int ny) {
  unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned idx = iy * nx + ix;

  if (ix < nx && iy < ny) {
    c[idx] = a[idx] + b[idx];
  }
}

using namespace std;

int main(int argc, char *argv[]) {
  int dev = 0;
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, dev));
  CHECK(cudaSetDevice(dev));

  const size_t nx = 1 << 14;
  const size_t ny = 1 << 14;
  const size_t nxy = nx * ny;
  const size_t nBytes = nxy * sizeof(float);

  vector<float> h_a(nxy), h_b(nxy), host_ref(nxy, 0), gpu_ref(nxy, 0);

  Timer timer;
  timer.reset();
  sumMatrixOnHost(h_a.data(), h_b.data(), host_ref.data(), nx, ny);

  // malloc device global memory
  float *d_a, *d_b, *d_c;
  CHECK(cudaMalloc((void **)&d_a, nBytes));
  CHECK(cudaMalloc((void **)&d_b, nBytes));
  CHECK(cudaMalloc((void **)&d_c, nBytes));

  // transfer data from host to device
  CHECK(cudaMemcpy(d_a, h_a.data(), nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, h_b.data(), nBytes, cudaMemcpyHostToDevice));

  int dimx = 32, dimy = 32;
  if (argc > 2) {
    dimx = atoi(argv[1]);
    dimy = atoi(argv[2]);
  }
  dim3 block(dimx, dimy);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  CHECK(cudaDeviceSynchronize());
  timer.reset();
  sumMatrixOnGPU2D<<<grid, block>>>(d_a, d_b, d_c, nx, ny);
  CHECK(cudaDeviceSynchronize());
  double elapsed = timer.elapsed_nanoseconds() / 1000000.0f;
  printf("sumMatrixOnGPU2D<<<(%d, %d), (%d, %d)>>> elapsed %lf ms\n",
         grid.x, grid.y, block.x, block.y, elapsed);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(gpu_ref.data(), d_c, nBytes, cudaMemcpyDeviceToHost));
  checkResult(host_ref.data(), gpu_ref.data(), nxy);

  CHECK(cudaFree(d_a));
  CHECK(cudaFree(d_b));
  CHECK(cudaFree(d_c));

  CHECK(cudaDeviceReset());
  return 0;
}
