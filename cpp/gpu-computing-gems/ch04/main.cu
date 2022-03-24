#include "timer.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <vector>

using namespace std;

static void gen_data(float *ax, float *ay, float *az, float *gx, float *gy,
                     float *gz, float *charge, float *size, const int atom,
                     const int grid) {
  const auto rnd_max = static_cast<float>(RAND_MAX);

  printf("Generating Data 1\n");
  for (int i = 0; i < atom; i++) {
    ax[i] = float(rand()) / rnd_max;
    ay[i] = float(rand()) / rnd_max;
    az[i] = float(rand()) / rnd_max;
    charge[i] = float(rand()) / rnd_max;
    size[i] = float(rand()) / rnd_max;
  }
  printf("Generating Data 2\n");
  for (int i = 0; i < grid; i++) {
    gx[i] = float(rand()) / rnd_max;
    gy[i] = float(rand()) / rnd_max;
    gz[i] = float(rand()) / rnd_max;
  }
  printf("Done generating inputs.\n\n");
}

static void mdh_scalar(const int grids, const int atoms, const float *ax,
                       const float *ay, const float *az, const float *gx,
                       const float *gy, const float *gz, const float *charge,
                       const float *size, const float xkappa, const float pre1,
                       float *val) {
  float dist;
  float dx, dy, dz;

  for (int i = 0; i < grids; i++) {
    for (int j = 0; j < atoms; j++) {
      dx = gx[i] - ax[j];
      dy = gy[i] - ay[j];
      dz = gz[i] - az[j];
      dist = sqrtf(dx * dx + dy * dy + dz * dz);
      val[i] += charge[j] * expf(-xkappa * (dist - size[j])) /
                (dist * (1.0f + xkappa * size[j]));
    }
    val[i] *= pre1;
  }
}

__global__ static void mdh(const float *ax, const float *ay, const float *az,
                           const float *gx, const float *gy, const float *gz,
                           const float *charge, const float *size, float *val,
                           const float pre1, const float xkappa, int atoms) {
  extern __shared__ float smem[];

  float v = 0.0f;
  int igrid = blockIdx.x * blockDim.x + threadIdx.x;
  const int csize = blockDim.x;
  int upper = csize;
  const int id = threadIdx.x;
  const float lgx = gx[igrid];
  const float lgy = gy[igrid];
  const float lgz = gz[igrid];

  for (int j = 0; j < atoms; j += csize) {
    __syncthreads();

    if (j + id < atoms) {
      smem[id] = ax[j + id];
      smem[id + csize] = ay[j + id];
      smem[id + 2 * csize] = az[j + id];
      smem[id + 3 * csize] = charge[j + id];
      smem[id + 4 * csize] = size[j + id];
    }

    __syncthreads();

    if (j + csize >= atoms) {
      upper = atoms - j;
    }

    for (int i = 0; i < upper; i++) {
      float dx = lgx - smem[i];
      float dy = lgy - smem[i + csize];
      float dz = lgz - smem[i + 2 * csize];
      float q = smem[i + 3 * csize];
      float sz = smem[i + 4 * csize];
      float dist = sqrtf(dx * dx + dy * dy + dz * dz);
      v += q * expf(-xkappa * (dist - sz)) / (dist * (1.0f + xkappa * sz));
    }
  }

  val[igrid] = pre1 * v;
}

static void test_mdh() {
  const int natom = 5877;
  const int ngrid = 134918;
  const int ngadj = ngrid + (512 - (ngrid & 511));
  const float pre1 = 4.46184985145e19f;
  const float xkappa = 0.0735516324639f;

  vector<float> ax(natom), ay(natom), az(natom), charge(natom), size(natom);
  vector<float> gx(ngadj), gy(ngadj), gz(ngadj), val1(ngadj), val2(ngadj);
  gen_data(ax.data(), ay.data(), az.data(), gx.data(), gy.data(), gz.data(),
           charge.data(), size.data(), natom, ngrid);

  std::fill(std::begin(val1), std::end(val1), 0.0f);
  std::fill(std::begin(val2), std::end(val2), 0.0f);

  Timer t;
  t.reset();
  mdh_scalar(ngrid, natom, ax.data(), ay.data(), az.data(), gx.data(),
             gy.data(), gz.data(), charge.data(), size.data(), xkappa, pre1,
             val1.data());
  printf("cpu costs  %.2lf ms\n", t.elapsed_milliseconds());

  thrust::device_vector<float> d_ax = ax, d_ay = ay, d_az = az;
  thrust::device_vector<float> d_gx = gx, d_gy = gy, d_gz = gz;
  thrust::device_vector<float> d_charge = charge, d_size = size;
  thrust::device_vector<float> d_val = val2;
  const float *d_ax_ptr = thrust::raw_pointer_cast(d_ax.data());
  const float *d_ay_ptr = thrust::raw_pointer_cast(d_ay.data());
  const float *d_az_ptr = thrust::raw_pointer_cast(d_az.data());
  const float *d_gx_ptr = thrust::raw_pointer_cast(d_gx.data());
  const float *d_gy_ptr = thrust::raw_pointer_cast(d_gy.data());
  const float *d_gz_ptr = thrust::raw_pointer_cast(d_gz.data());
  const float *d_charge_ptr = thrust::raw_pointer_cast(d_charge.data());
  const float *d_size_ptr = thrust::raw_pointer_cast(d_size.data());
  float *d_val_ptr = thrust::raw_pointer_cast(d_val.data());
  const int threadsPerBlock = 64;

  t.reset();
  mdh<<<(ngrid + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock,
        sizeof(float) * threadsPerBlock * 5>>>(
      d_ax_ptr, d_ay_ptr, d_az_ptr, d_gx_ptr, d_gy_ptr, d_gz_ptr, d_charge_ptr,
      d_size_ptr, d_val_ptr, pre1, xkappa, natom);

  cudaDeviceSynchronize();
  printf("gpus cost %.2f ms\n", t.elapsed_milliseconds());

  cudaMemcpy(val2.data(), d_val_ptr, sizeof(float) * val2.size(),
             cudaMemcpyDeviceToHost);

  const float E = 0.1f * pre1;
  for (auto i = 0; i < ngrid; i++) {
    if (fabsf(val1[i] - val2[i]) > E) {
      printf("[%d] cpu = %.2f, gpu = %.2f\n", i, val1[i], val2[i]);
      exit(-1);
    }
  }
}

int main() { return 0; }
