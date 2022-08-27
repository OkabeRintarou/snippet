#include "common.h"
#include "timer.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

int recursiveReduce(int *data, const int n) {
  if (n == 1) {
    return data[0];
  }
  const int stride = n / 2;

  for (int i = 0; i < stride; i++) {
    data[i] += data[i + stride];
  }
  return recursiveReduce(data, stride);
}

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned n) {
  unsigned tid = threadIdx.x;
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  // convert global data pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x;

  if (idx >= n)
    return;

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0) {
      idata[tid] += idata[tid + stride];

      __syncthreads();
    }
  }
  // write result for this result to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned n) {
  unsigned tid = threadIdx.x;
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  // convert global data pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x;

  if (idx >= n)
    return;

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }

    __syncthreads();
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned n) {
  unsigned tid = threadIdx.x;
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x;

  if (idx >= n)
    return;

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int index = 2 * stride * tid;
    if (index < blockDim.x) {
      idata[index] += idata[index + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned n) {
  unsigned tid = threadIdx.x;
  unsigned idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x * 2;

  // unrolling 2 data blocks
  if (idx + blockDim.x < n) {
    g_idata[idx] += g_idata[idx + blockDim.x];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}
__global__ void reduceUnrolling4(int *g_idata, int *g_odata, unsigned n) {
  unsigned tid = threadIdx.x;
  unsigned idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x * 4;

  // unrolling 4 data blocks
  if (idx + 3 * blockDim.x < n) {
    g_idata[idx] += g_idata[idx + blockDim.x] + g_idata[idx + 2 * blockDim.x] +
                    g_idata[idx + 3 * blockDim.x];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned n) {
  unsigned tid = threadIdx.x;
  unsigned idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x * 8;

  // unrolling 8 data blocks
  if (idx + 7 * blockDim.x < n) {
    g_idata[idx] +=
        g_idata[idx + blockDim.x] + g_idata[idx + 2 * blockDim.x] +
        g_idata[idx + 3 * blockDim.x] + g_idata[idx + 4 * blockDim.x] +
        g_idata[idx + 5 * blockDim.x] + g_idata[idx + 6 * blockDim.x] +
        g_idata[idx + 7 * blockDim.x];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceUnrollWarps8(int *g_idata, int *g_odata, unsigned n) {
  unsigned tid = threadIdx.x;
  unsigned idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x * 8;

  // unrolling 8 data blocks
  if (idx + 7 * blockDim.x < n) {
    g_idata[idx] +=
        g_idata[idx + blockDim.x] + g_idata[idx + 2 * blockDim.x] +
        g_idata[idx + 3 * blockDim.x] + g_idata[idx + 4 * blockDim.x] +
        g_idata[idx + 5 * blockDim.x] + g_idata[idx + 6 * blockDim.x] +
        g_idata[idx + 7 * blockDim.x];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }
    __syncthreads();
  }

  if (tid < 32) {
    volatile int *vmem = idata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
  }
  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceCompleteUnrollWarps8(int *g_idata, int *g_odata, unsigned n) {
  unsigned tid = threadIdx.x;
  unsigned idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x * 8;

  // unrolling 8 data blocks
  if (idx + 7 * blockDim.x < n) {
    g_idata[idx] +=
        g_idata[idx + blockDim.x] + g_idata[idx + 2 * blockDim.x] +
        g_idata[idx + 3 * blockDim.x] + g_idata[idx + 4 * blockDim.x] +
        g_idata[idx + 5 * blockDim.x] + g_idata[idx + 6 * blockDim.x] +
        g_idata[idx + 7 * blockDim.x];
  }
  __syncthreads();

  if (blockDim.x >= 1024 && tid < 512) {
    idata[tid] += idata[tid + 512];
  }
  __syncthreads();

  if (blockDim.x >= 512 && tid < 256) {
    idata[tid] += idata[tid + 256];
  }
  __syncthreads();

  if (blockDim.x >= 256 && tid < 128) {
    idata[tid] += idata[tid + 128];
  }
  __syncthreads();

  if (blockDim.x >= 128 && tid < 64) {
    idata[tid] += idata[tid + 64];
  }
  __syncthreads();

  if (tid < 32) {
    volatile int *vmem = idata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
  }
  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

using namespace std;

int main(int argc, char *argv[]) {
  int dev = 0;
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, dev);
  printf("device %d: %s ", dev, prop.name);
  cudaSetDevice(dev);

  const size_t size = 1 << 24;
  printf("\tWith array size %lu\n", size);
  const size_t bytes = size * sizeof(int);

  int blockSize = 512;
  if (argc > 1) {
    blockSize = atoi(argv[1]);
  }

  dim3 block(blockSize, 1);
  dim3 grid((size + block.x - 1) / block.x, 1);
  printf("grid %d, block %d\n", grid.x, block.x);

  vector<int> h_idata(size), h_odata(grid.x);
  for (size_t i = 0; i < size; i++) {
    h_idata[i] = (int)(rand() & 0xff);
  }
  vector<int> tmp = h_idata;

  int *d_idata, *d_odata;
  CHECK(cudaMalloc((void **)&d_idata, bytes));
  CHECK(cudaMalloc((void **)&d_odata, grid.x * sizeof(int)));

  Timer timer;
  timer.reset();
  // cpu reduction
  int cpu_sum = recursiveReduce(tmp.data(), size);
  printf("cpu reduce elapsed %f ms cpu_sum: %d\n",
         timer.elapsed_microseconds() / 1000.0, cpu_sum);

  // warm up
  CHECK(cudaMemcpy(d_idata, h_idata.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_odata, h_odata.data(), grid.x * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  timer.reset();
  reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
  CHECK(cudaDeviceSynchronize());

  // kernel 1: reduceNeighbored
  CHECK(cudaMemcpy(d_idata, h_idata.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_odata, h_odata.data(), grid.x * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  timer.reset();
  reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
  CHECK(cudaDeviceSynchronize());
  double elapsed1 = timer.elapsed_microseconds() / 1000.0;
  CHECK(cudaMemcpy(h_odata.data(), d_odata, grid.x * sizeof(int),
                   cudaMemcpyDeviceToHost));
  int gpu_sum = 0;
  for (int i = 0; i < grid.x; i++) {
    gpu_sum += h_odata[i];
  }
  printf("gpu Neighbored elapsed %f ms gpu_sum: %d\n", elapsed1, gpu_sum);

  // kernel 2: reduceNeighboredLess
  CHECK(cudaMemcpy(d_idata, h_idata.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_odata, h_odata.data(), grid.x * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  timer.reset();
  reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
  CHECK(cudaDeviceSynchronize());
  double elapsed2 = timer.elapsed_microseconds() / 1000.0;
  CHECK(cudaMemcpy(h_odata.data(), d_odata, grid.x * sizeof(int),
                   cudaMemcpyDeviceToHost));
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++) {
    gpu_sum += h_odata[i];
  }
  printf("gpu NeighboredLess elapsed %f ms gpu_sum: %d\n", elapsed2, gpu_sum);

  // kernel 3: reduceInterleaved
  CHECK(cudaMemcpy(d_idata, h_idata.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_odata, h_odata.data(), grid.x * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  timer.reset();
  reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
  CHECK(cudaDeviceSynchronize());
  double elapsed3 = timer.elapsed_microseconds() / 1000.0;
  CHECK(cudaMemcpy(h_odata.data(), d_odata, grid.x * sizeof(int),
                   cudaMemcpyDeviceToHost));
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++) {
    gpu_sum += h_odata[i];
  }
  printf("gpu Interleaved elapsed %f ms gpu_sum: %d\n", elapsed3, gpu_sum);

  // kernel 4: reduceUnrolling2
  CHECK(cudaMemcpy(d_idata, h_idata.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_odata, h_odata.data(), grid.x * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  timer.reset();
  reduceUnrolling2<<<grid.x / 2, block>>>(d_idata, d_odata, size);
  CHECK(cudaDeviceSynchronize());
  double elapsed4 = timer.elapsed_microseconds() / 1000.0;
  CHECK(cudaMemcpy(h_odata.data(), d_odata, grid.x * sizeof(int),
                   cudaMemcpyDeviceToHost));
  gpu_sum = 0;
  for (int i = 0; i < grid.x / 2; i++) {
    gpu_sum += h_odata[i];
  }
  printf("gpu Unrolling2 elapsed %f ms gpu_sum: %d\n", elapsed4, gpu_sum);

  // kernel 5: reduceUnrolling4
  CHECK(cudaMemcpy(d_idata, h_idata.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_odata, h_odata.data(), grid.x * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  timer.reset();
  reduceUnrolling4<<<grid.x / 4, block>>>(d_idata, d_odata, size);
  CHECK(cudaDeviceSynchronize());
  double elapsed5 = timer.elapsed_microseconds() / 1000.0;
  CHECK(cudaMemcpy(h_odata.data(), d_odata, grid.x * sizeof(int),
                   cudaMemcpyDeviceToHost));
  gpu_sum = 0;
  for (int i = 0; i < grid.x / 4; i++) {
    gpu_sum += h_odata[i];
  }
  printf("gpu Unrolling4 elapsed %f ms gpu_sum: %d\n", elapsed5, gpu_sum);

  // kernel 6: reduceUnrolling8
  CHECK(cudaMemcpy(d_idata, h_idata.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_odata, h_odata.data(), grid.x * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  timer.reset();
  reduceUnrolling8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
  CHECK(cudaDeviceSynchronize());
  double elapsed6 = timer.elapsed_microseconds() / 1000.0;
  CHECK(cudaMemcpy(h_odata.data(), d_odata, grid.x * sizeof(int),
                   cudaMemcpyDeviceToHost));
  gpu_sum = 0;
  for (int i = 0; i < grid.x / 8; i++) {
    gpu_sum += h_odata[i];
  }
  printf("gpu Unrolling8 elapsed %f ms gpu_sum: %d\n", elapsed6, gpu_sum);

  // kernel 7: reduceUnrollWarps8
  CHECK(cudaMemcpy(d_idata, h_idata.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_odata, h_odata.data(), grid.x * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  timer.reset();
  reduceUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
  CHECK(cudaDeviceSynchronize());
  double elapsed7 = timer.elapsed_microseconds() / 1000.0;
  CHECK(cudaMemcpy(h_odata.data(), d_odata, grid.x * sizeof(int),
                   cudaMemcpyDeviceToHost));
  gpu_sum = 0;
  for (int i = 0; i < grid.x / 8; i++) {
    gpu_sum += h_odata[i];
  }
  printf("gpu UnrollWarps8 elapsed %f ms gpu_sum: %d\n", elapsed7, gpu_sum);

  // kernel : reduceCompleteUnrollWarps8
  CHECK(cudaMemcpy(d_idata, h_idata.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_odata, h_odata.data(), grid.x * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  timer.reset();
  reduceCompleteUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
  CHECK(cudaDeviceSynchronize());
  double elapsed8 = timer.elapsed_microseconds() / 1000.0;
  CHECK(cudaMemcpy(h_odata.data(), d_odata, grid.x * sizeof(int),
                   cudaMemcpyDeviceToHost));
  gpu_sum = 0;
  for (int i = 0; i < grid.x / 8; i++) {
    gpu_sum += h_odata[i];
  }
  printf("gpu CompleteUnrollWarps8 elapsed %f ms gpu_sum: %d\n", elapsed8, gpu_sum);
  return 0;
}
