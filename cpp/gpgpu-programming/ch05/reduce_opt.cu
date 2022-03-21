#include "timer.h"
#include "util.h"
#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>
#include <thread>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <vector>

static const int block_num = 32;
static const int thread_num = 256;
static const int DEFAULT_DATA_SIZE = 4096000;

static void verify_output(double expect, double actual) {
  const double E = 0.0001;
  if (std::fabs(expect - actual) > E) {
    printf("Error, expect: %.2lf actual: %.2lf\n", expect, actual);
  }
}

class GPUTimer {
public:
  GPUTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&end);
  }
  void reset() { cudaEventRecord(start, 0); }
  float elapsed() {
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, start, end);
    return elapsed;
  }
  ~GPUTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(end);
  }

private:
  cudaEvent_t start, end;
};

double cpu_serialize(const double *input, double *output, int size) {
  Timer timer;
  double sum = 0.0;
  timer.reset();

  for (int i = 0; i < size; i++) {
    sum += input[i] * input[i];
  }
  *output = sum;
  return timer.elapsed_milliseconds();
}

double cpu_parallel(double *input, double *output, int size) {
  int thread_count = 12;
#ifdef THREAD_COUNT
  thread_count = THREAD_COUNT;
#endif
  auto thread_func = [](double *in, double *out, int start, int end) {
    double sum = 0.0;
    for (int i = start; i < end; i++) {
      sum += in[i] * in[i];
    }
    *out = sum;
  };

  std::vector<double> sums(thread_count, 0.0);
  std::vector<std::thread> threads;
  const int step = size / thread_count;

  Timer timer;
  timer.reset();
  for (int i = 0; i < thread_count; i++) {
    int start = i * step;
    int end = std::min(start + step, size);
    if ((i == thread_count - 1) && end < size) {
      end = size;
    }
    threads.emplace_back(thread_func, input, &sums[i], start, end);
  }

  for (auto &&t : threads) {
    t.join();
  }

  double sum = 0.0;
  for (double r : sums) {
    sum += r;
  }
  *output = sum;
  return timer.elapsed_milliseconds();
}

__global__ static void gpu_sum_serialize(const double *input, double *output,
                                         int size) {
  double sum = 0.0;
  for (int i = 0; i < size; i++) {
    sum += input[i] * input[i];
  }
  *output = sum;
}

double gpu_sum_serialize_host(double *input, double *output, int size) {
  double *d_input, *d_output;

  cudaMalloc((void **)&d_input, sizeof(double) * size);
  cudaMalloc((void **)&d_output, sizeof(double));
  cudaMemcpy(d_input, input, sizeof(double) * size, cudaMemcpyHostToDevice);

  GPUTimer timer;
  timer.reset();
  gpu_sum_serialize<<<1, 1, 0>>>(d_input, d_output, size);
  double elapsed = timer.elapsed();

  cudaMemcpy(output, d_output, sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(d_input);
  cudaFree(d_output);
  return elapsed;
}

__global__ static void gpu_sum_parallel_1x256(const double *input,
                                              double *output, int size) {
  const int tid = threadIdx.x;
  int step = size / thread_num;
  if (size % thread_num) {
    ++step;
  }
  const int start = tid * step;
  const int end = start + step;
  double sum = 0.0;

  for (int i = start; i < end; i++) {
    if (i < size) {
      sum += input[i] * input[i];
    }
  }
  output[tid] = sum;
}

double gpu_sum_parallel_1x256_host(double *input, double *output, int size) {
  double *d_input, *d_output;
  std::vector<double> h_output(thread_num, 0.0);

  cudaMalloc((void **)&d_input, sizeof(double) * size);
  cudaMalloc((void **)&d_output, sizeof(double) * thread_num);
  cudaMemcpy(d_input, input, sizeof(double) * size, cudaMemcpyHostToDevice);

  GPUTimer timer;
  timer.reset();
  gpu_sum_parallel_1x256<<<1, thread_num>>>(d_input, d_output, size);
  double elapsed = timer.elapsed();

  cudaMemcpy(h_output.data(), d_output, sizeof(double) * thread_num,
             cudaMemcpyDeviceToHost);
  double sum = 0.0;
  for (int i = 0; i < thread_num; i++) {
    sum += h_output[i];
  }

  *output = sum;
  cudaFree(d_input);
  cudaFree(d_output);
  return elapsed;
}

__global__ static void
gpu_sum_parallel_1x256_bank_opt(const double *input, double *output, int size) {
  const int tid = threadIdx.x;
  double sum = 0.0;

  for (int i = tid; i < size; i += blockDim.x) {
    sum += input[i] * input[i];
  }
  output[tid] = sum;
}

double gpu_sum_parallel_1x256_bank_opt_host(double *input, double *output,
                                            int size) {
  double *d_input, *d_output;
  std::vector<double> h_output(thread_num, 0.0f);

  cudaMalloc((void **)&d_input, sizeof(double) * size);
  cudaMalloc((void **)&d_output, sizeof(double) * thread_num);
  cudaMemcpy(d_input, input, sizeof(double) * size, cudaMemcpyHostToDevice);

  GPUTimer timer;
  timer.reset();
  gpu_sum_parallel_1x256_bank_opt<<<1, thread_num>>>(d_input, d_output, size);
  double elapsed = timer.elapsed();

  cudaDeviceSynchronize();
  cudaMemcpy(h_output.data(), d_output, sizeof(double) * thread_num,
             cudaMemcpyDeviceToHost);
  double sum = 0.0;

  for (int i = 0; i < thread_num; i++) {
    sum += h_output[i];
  }

  *output = sum;
  cudaFree(d_input);
  cudaFree(d_output);
  return elapsed;
}

__global__ static void gpu_sum_parallel_32x256(const double *input,
                                               double *output, int size) {

  const int bid = blockIdx.x, tid = threadIdx.x;
  const int start = bid * blockDim.x + tid;
  const int step = blockDim.x * gridDim.x;
  double sum = 0.0;

  for (int i = start; i < size; i += step) {
    sum += input[i] * input[i];
  }
  output[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

double gpu_sum_parallel_32x256_host(double *input, double *output, int size) {
  double *d_input, *d_output;
  std::vector<double> h_output(thread_num * block_num, 0.0);

  cudaMalloc((void **)&d_input, sizeof(double) * size);
  cudaMalloc((void **)&d_output, sizeof(double) * thread_num * block_num);
  cudaMemcpy(d_input, input, sizeof(double) * size, cudaMemcpyHostToDevice);

  GPUTimer timer;
  timer.reset();
  gpu_sum_parallel_32x256<<<block_num, thread_num>>>(d_input, d_output, size);
  double elapsed = timer.elapsed();

  cudaMemcpy(h_output.data(), d_output, sizeof(double) * thread_num * block_num,
             cudaMemcpyDeviceToHost);
  double sum = 0.0;
  for (int i = 0; i < thread_num * block_num; i++) {
    sum += h_output[i];
  }

  *output = sum;
  cudaFree(d_input);
  cudaFree(d_output);
  return elapsed;
}

__global__ static void gpu_sum_parallel_32x256_shmem_opt(const double *input,
                                                         double *output,
                                                         int size) {
  __shared__ int shared[thread_num];

  int tid = threadIdx.x;
  double sum = 0.0;
  shared[tid] = 0.0;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    sum += input[i] * input[i];
  }

  shared[tid] = sum;

  __syncthreads();

  int offset = blockDim.x / 2;
  while (offset > 0) {
    if (tid < offset) {
      shared[tid] += shared[tid + offset];
    }
    offset >>= 1;
    __syncthreads();
  }
  if (tid == 0) {
    output[blockIdx.x] = shared[0];
  }
}

double gpu_sum_parallel_32x256_shmem_opt_host(double *input, double *output,
                                              int size) {
  double *d_input, *d_output;
  std::vector<double> h_output(block_num, 0.0);

  cudaMalloc((void **)&d_input, sizeof(double) * size);
  cudaMalloc((void **)&d_output, sizeof(double) * block_num);
  cudaMemcpy(d_input, input, sizeof(double) * size, cudaMemcpyHostToDevice);

  GPUTimer timer;
  timer.reset();
  gpu_sum_parallel_32x256_shmem_opt<<<block_num, thread_num>>>(d_input,
                                                               d_output, size);
  double elapsed = timer.elapsed();
  cudaMemcpy(h_output.data(), d_output, sizeof(double) * block_num,
             cudaMemcpyDeviceToHost);
  double sum = 0.0;
  for (int i = 0; i < block_num; i++) {
    sum += h_output[i];
  }

  *output = sum;
  cudaFree(d_input);
  cudaFree(d_output);
  return elapsed;
}

__global__ static void gpu_sum_parallel_32x256_inst_opt(const double *input,
                                                        double *output,
                                                        int size) {
  __shared__ int shared[thread_num];

  int tid = threadIdx.x;
  double sum = 0.0;
  shared[tid] = 0.0;

#pragma unroll
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += __mul24(blockDim.x, gridDim.x)) {
    sum += input[i] * input[i];
  }

  shared[tid] = sum;

  __syncthreads();

  if (tid < 128) {
    shared[tid] += shared[tid + 128];
  }
  __syncthreads();
  if (tid < 64) {
    shared[tid] += shared[tid + 64];
  }
  __syncthreads();
  if (tid < 32) {
    shared[tid] += shared[tid + 32];
  }
  __syncthreads();
  if (tid < 16) {
    shared[tid] += shared[tid + 16];
  }
  __syncthreads();
  if (tid < 8) {
    shared[tid] += shared[tid + 8];
  }
  __syncthreads();
  if (tid < 4) {
    shared[tid] += shared[tid + 4];
  }
  __syncthreads();
  if (tid < 2) {
    shared[tid] += shared[tid + 2];
  }
  __syncthreads();
  if (tid < 1) {
    shared[tid] += shared[tid + 1];
  }
  __syncthreads();

  if (tid == 0) {
    output[blockIdx.x] = shared[0];
  }
}

double gpu_sum_parallel_32x256_inst_opt_host(double *input, double *output,
                                             int size) {
  double *d_input, *d_output;
  std::vector<double> h_output(block_num, 0.0);

  cudaMalloc((void **)&d_input, sizeof(double) * size);
  cudaMalloc((void **)&d_output, sizeof(double) * block_num);
  cudaMemcpy(d_input, input, sizeof(double) * size, cudaMemcpyHostToDevice);

  GPUTimer timer;
  timer.reset();
  gpu_sum_parallel_32x256_inst_opt<<<block_num, thread_num>>>(d_input, d_output,
                                                              size);
  double elapsed = timer.elapsed();
  cudaMemcpy(h_output.data(), d_output, sizeof(double) * block_num,
             cudaMemcpyDeviceToHost);
  double sum = 0.0;
  for (int i = 0; i < block_num; i++) {
    sum += h_output[i];
  }

  *output = sum;
  cudaFree(d_input);
  cudaFree(d_output);
  return elapsed;
}

struct square {
  __host__ __device__ double operator()(double x) { return x * x; }
};

double thrust_host(double *input, double *output, int size) {

  double *d_input;
  std::vector<double> h_output(block_num, 0.0);
  cudaMalloc((void **)&d_input, sizeof(double) * size);
  cudaMemcpy(d_input, input, sizeof(double) * size, cudaMemcpyHostToDevice);

  Timer timer;
  timer.reset();
  thrust::device_ptr<double> dev_ptr(d_input);
  *output = thrust::transform_reduce(dev_ptr, dev_ptr + size, square(), 0.0,
                                     thrust::plus<double>());
  double elapsed = timer.elapsed_milliseconds();
  cudaFree(d_input);
  return elapsed;
}

int main() {
  int data_size = DEFAULT_DATA_SIZE;
  const int iter_times = 5;
#ifdef DATA_SIZE
  data_size = DATA_SIZE;
#endif
  auto min_time = DBL_MAX;
  double standard_sum;

  std::vector<double> input(data_size);
  double output = 0.0;

  generateNumbers(input.data(), data_size);

  printf("===============\n");

  // cpu serialize
  for (int i = 0; i < iter_times; i++) {
    double elapsed = cpu_serialize(input.data(), &output, data_size);
    min_time = std::min(min_time, elapsed);
  }
  standard_sum = output;
  printf("cpu serialize: %.2lf ms\n", min_time);

  min_time = DBL_MAX;
  output = 0.0f;

  // cpu parallel
  for (int i = 0; i < iter_times; i++) {
    double elapsed = cpu_parallel(input.data(), &output, data_size);
    min_time = std::min(min_time, elapsed);
  }
  verify_output(standard_sum, output);
  printf("cpu parallel: %.2f ms\n", min_time);

  min_time = DBL_MAX;
  output = 0.0f;

  // gpu serialize
  for (int i = 0; i < iter_times; i++) {
    double elapsed = gpu_sum_serialize_host(input.data(), &output, data_size);
    min_time = std::min(min_time, elapsed);
  }
  verify_output(standard_sum, output);
  printf("gpu serialize: %.2f ms\n", min_time);

  min_time = DBL_MAX;
  output = 0.0f;

  // gpu parallel: one thread block, each block 256 threads
  for (int i = 0; i < iter_times; i++) {
    double elapsed =
        gpu_sum_parallel_1x256_host(input.data(), &output, data_size);
    min_time = std::min(min_time, elapsed);
  }
  verify_output(standard_sum, output);
  printf("gpu parallel(1x256): %.2f ms\n", min_time);
  min_time = DBL_MAX;
  output = 0.0f;

  // gpu parallel: avoid bank conflict
  for (int i = 0; i < iter_times; i++) {
    double elapsed =
        gpu_sum_parallel_1x256_bank_opt_host(input.data(), &output, data_size);
    min_time = std::min(min_time, elapsed);
  }
  verify_output(standard_sum, output);
  printf("gpu parallel(1x256) bank opt: %.2f ms\n", min_time);
  min_time = DBL_MAX;
  output = 0.0f;

  // gpu parallel: 32 thread blocks, each block 256 threads
  for (int i = 0; i < iter_times; i++) {
    double elapsed =
        gpu_sum_parallel_32x256_host(input.data(), &output, data_size);
    min_time = std::min(min_time, elapsed);
  }
  verify_output(standard_sum, output);
  printf("gpu parallel(32x256): %.2f ms\n", min_time);
  min_time = DBL_MAX;
  output = 0.0f;

  // gpu parallel: using share memory
  for (int i = 0; i < iter_times; i++) {
    double elapsed = gpu_sum_parallel_32x256_shmem_opt_host(input.data(),
                                                            &output, data_size);
    min_time = std::min(min_time, elapsed);
  }
  verify_output(standard_sum, output);
  printf("gpu parallel(32x256) shmem opt: %.2f ms\n", min_time);
  min_time = DBL_MAX;

  // gpu parallel: using share memory
  for (int i = 0; i < iter_times; i++) {
    double elapsed =
        gpu_sum_parallel_32x256_inst_opt_host(input.data(), &output, data_size);
    min_time = std::min(min_time, elapsed);
  }
  verify_output(standard_sum, output);
  printf("gpu parallel(32x256) inst opt: %.2f ms\n", min_time);
  min_time = DBL_MAX;
  output = 0.0f;
  output = 0.0f;

  // thrust
  for (int i = 0; i < iter_times; i++) {
    double elapsed = thrust_host(input.data(), &output, data_size);
    min_time = std::min(min_time, elapsed);
  }
  verify_output(standard_sum, output);
  printf("thrust: %.2f ms\n", min_time);
  return 0;
}