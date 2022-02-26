#include "common.h"
#include "cpu_bitmap.h"
#include <cmath>

static constexpr float INF = 2e10f;
static constexpr unsigned SPHERES = 20u;
static constexpr int DIM = 1024;

struct Sphere {
  float r, g, b;
  float radius;
  float x, y, z;

  // given a ray shot from the pixel at (ox, oy),
  // hit() computes whether the ray intersects the sphere.
  // if the ray does intersect the sphere, return the distance
  // from the camera where the ray hits the sphere
  __device__ float hit(float ox, float oy, float *n) {
    float dx = ox - x;
    float dy = oy - y;
    if (dx * dx + dy * dy < radius * radius) {
      float dz = sqrtf( radius*radius - dx*dx - dy*dy );
      *n = dz / sqrtf( radius * radius );
      return dz + z;
    }
    return -INF;
  }
};

static inline float rnd(float x) {
  return x * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

__global__ void kernel(unsigned char *ptr, Sphere *s) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float ox = float(x - DIM / 2);
  float oy = float(y - DIM / 2);

  float r = .0f, g = .0f, b = .0f;
  float maxz = -INF;

  for (unsigned i = 0; i < SPHERES; i++) {
    float n;
    float t = s[i].hit(ox, oy, &n);
    if (t > maxz) {
      float fscale = n;
      r = s[i].r * fscale;
      g = s[i].g * fscale;
      b = s[i].b * fscale;
      maxz = t;
    }
  }

  ptr[offset * 4 + 0] = int(r * 255.0f);
  ptr[offset * 4 + 1] = int(g * 255.0f);
  ptr[offset * 4 + 2] = int(b * 255.0f);
  ptr[offset * 4 + 3] = 255;
}

int main() {

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start, 0));

  CPUBitmap bitmap(DIM, DIM);
  unsigned char *dev_bitmap;
  CHECK(cudaMalloc(&dev_bitmap, bitmap.image_size()));

  cuda::vector<Sphere> spheres(SPHERES);
  for (unsigned i = 0; i < SPHERES; i++) {
    Sphere *s = &spheres[i];
    s->r = rnd(1.0f);
    s->g = rnd(1.0f);
    s->b = rnd(1.0f);
    s->x = rnd(1000.0f) - 500.0f;
    s->y = rnd(1000.0f) - 500.0f;
    s->z = rnd(1000.0f) - 500.0f;
    s->radius = rnd(100.0f) + 20.0f;
  }

  dim3 grids(DIM / 16, DIM / 16);
  dim3 threads(16, 16);
  kernel<<<grids, threads>>>(dev_bitmap, spheres.data());

  CHECK(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(),
                   cudaMemcpyDeviceToHost));

  CHECK(cudaEventRecord(stop, 0));
  CHECK(cudaEventSynchronize(stop));
  float elapsed_time = .0f;
  CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
  printf("Time to generate: %3.1f ms\n", elapsed_time);
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));

  bitmap.display_and_exit();

  CHECK(cudaFree(dev_bitmap));
  return 0;
}
