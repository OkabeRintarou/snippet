#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <hip/hip_runtime.h>

static const int WIDTH = 1024;
static const int HEIGHT = 1024;
static const int NUM = WIDTH * HEIGHT;

static const int THREAD_PER_BLOCK_X = 16;
static const int THREAD_PER_BLOCK_Y = 16;
static const int THREAD_PER_BLOCK_Z = 1;

__global__ void
vector_add(float *a, const float *b, const float *c, int width, int height) {
	int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

	int i = y * width + x;
	if (i < NUM)
		a[i] = b[i] + c[i];
}

using namespace std;

#define HIP_ASSERT(x) (assert((x) == hipSuccess))

int main() {
	float *hostA, *hostB, *hostC;
	float *deviceA, *deviceB, *deviceC;

	hipDeviceProp_t dev_prop;
	hipGetDeviceProperties(&dev_prop, 0);
	cout << " System minor " << dev_prop.minor << endl;
	cout << " System major " << dev_prop.major << endl;
	cout << " agent prop name " << dev_prop.name << endl;

	hostA = new float[NUM];
	hostB = new float[NUM];
	hostC = new float[NUM];

	for (int i = 0; i < NUM; i++) {
		hostB[i] = (float)i;
		hostC[i] = (float)i * 100.0f;
	}

	HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(float)));
	HIP_ASSERT(hipMalloc((void**)&deviceB, NUM * sizeof(float)));
	HIP_ASSERT(hipMalloc((void**)&deviceC, NUM * sizeof(float)));

	HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM * sizeof(float), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemcpy(deviceC, hostC, NUM * sizeof(float), hipMemcpyHostToDevice));

	hipLaunchKernelGGL(vector_add,
			dim3(WIDTH/THREAD_PER_BLOCK_X, HEIGHT/THREAD_PER_BLOCK_Y),
			dim3(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y),
			0, 0,
			deviceA, deviceB, deviceC, WIDTH, HEIGHT);

	HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM * sizeof(float), hipMemcpyDeviceToHost));

	// verify the results
	int errors = 0;
	for (int i = 0; i < NUM; i++) {
		if (hostA[i] != (hostB[i] + hostC[i])) {
			++errors;
		}
	}

	if (errors != 0) {
		printf("FAILED: %d errors\n", errors);
	} else {
		printf("PASSED!\n");
	}
	
	HIP_ASSERT(hipFree(deviceA));
	HIP_ASSERT(hipFree(deviceB));
	HIP_ASSERT(hipFree(deviceC));

	delete[] hostA;
	delete[] hostB;
	delete[] hostC;

	return 0;
}
