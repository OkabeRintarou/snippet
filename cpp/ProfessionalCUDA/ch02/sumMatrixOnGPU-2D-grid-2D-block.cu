#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK(call)					\
{									\
	const cudaError_t error = call;	\
	if (error != cudaSuccess)		\
	{								\
		printf("Error: %s:%d, ", __FILE__, __LINE__);				\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));	\
		exit(-10 * error);															\
	}																		\
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
	double epsilon = 1.0E-8;
	bool match = 1;
	for (int i = 0; i < N; i++) {
		if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
			match = 0;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}

	if (match) {
		printf("Always match.\n\n");
	}
}

void initialData(float *ip, int N) {
	time_t t;
	srand((unsigned int)(time(&t)));

	for (int i = 0; i < N; i++) {
		ip[i] = (float)(rand() & 0xff) / 10.0f;
	}
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny) {
	float *ia = A, *ib = B, *ic = C;

	for (int iy = 0; iy < ny; iy++) {
		for (int ix = 0; ix < nx; ix++) {
			ic[ix] = ia[ix] + ib[ix];
		}
		ia += nx; ib += nx; ic += nx;
	}
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, const int nx, const int ny) {
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;
	
	if (ix < nx && iy < ny) {
		MatC[idx] = MatA[idx] + MatB[idx];
	}
}

int main(int argc, char *argv[]) {
	printf("%s Starting...\n", argv[0]);

	// get device information
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	// set up data size of matrix
	int nx = 1 << 14;
	int ny = 1 << 14;
	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);
	printf("Matrix size: nx %d ny %d\n", nx, ny);

	// malloc host memory
	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float*)malloc(nBytes);
	h_B = (float*)malloc(nBytes);
	hostRef = (float*)malloc(nBytes);
	gpuRef = (float*)malloc(nBytes);

	// initialize matrix data at host
	initialData(h_A, nxy);
	initialData(h_B, nxy);
	memset(gpuRef, 0, nBytes);
	memset(hostRef, 0, nBytes);

	// add matrix at host side for check result
	sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);

	// malloc device memory
	float *d_MatA, *d_MatB, *d_MatC;
	cudaMalloc((void**)&d_MatA, nBytes);
	cudaMalloc((void**)&d_MatB, nBytes);
	cudaMalloc((void**)&d_MatC, nBytes);

	// transfer data from host to device
	cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

	// set up execution configuration
	int dimx = 32;
	int dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	// invoke the kernel
	sumMatrixOnGPU2D <<<grid, block>>> (d_MatA, d_MatB, d_MatC, nx, ny);
	cudaDeviceSynchronize();

	// copy kernel result back to host side
	cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

	checkResult(hostRef, gpuRef, nxy);

	// free host and device memory
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);
	cudaFree(d_MatA);
	cudaFree(d_MatB);
	cudaFree(d_MatC);

	// reset device
	cudaDeviceReset();

	return 0;
}
