#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>

#include <hip/hip_runtime.h>
#include "benchmark-utils.hpp"


using namespace std;

inline void hip_last_error_check() {
	hipError_t ec = hipGetLastError();
	if (ec != hipSuccess) {
		stringstream ss;
		ss << "HIP Runtime API error " << ec << ": " << hipGetErrorString(ec) << endl;
		throw runtime_error(ss.str());
	}
}

#define CHECK(c) \
	(c);		\
	hip_last_error_check();

template<typename T>
__global__ void
elementwise_add(const T *x,
				const T *y,
				T *z,
				unsigned stride,
				unsigned size) {
	for (unsigned i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
			i < size;
			i += hipGridDim_x * hipBlockDim_x) {
		z[i * stride] = x[i * stride] + y[i * stride];
	}
}

int main(int argc, char *argv[]) {

	hipDeviceProp_t prop;
	CHECK(hipGetDeviceProperties(&prop, 0));
	cout << "# Using device: " << prop.name << endl;
	
	using NumericT = float;
	const size_t N = 1000000;
	const size_t bytes = sizeof(NumericT) * N;
	vector<NumericT> host_x(32 * N);
	
	NumericT *x, *y, *z;
	CHECK(hipMalloc(&x, bytes * 32));
	CHECK(hipMalloc(&y, bytes * 32));
	CHECK(hipMalloc(&z, bytes * 32));

	// Warmup calculation
	hipLaunchKernelGGL(elementwise_add<NumericT>, dim3(256), dim3(256), 0, 0, x, y, z,
				static_cast<unsigned>(1),
				static_cast<unsigned>(N));

	Timer timer;
	cout << "# stride\ttime\tGB/sec" << endl;
	for (size_t stride = 0; stride <= 32; ++stride) {
		hipDeviceSynchronize();
		timer.start();
		for (size_t num_runs = 0; num_runs < 20; ++num_runs) {
			hipLaunchKernelGGL(elementwise_add<NumericT>, dim3(256), dim3(256), 0, 0, x, y, z,
						static_cast<unsigned>(stride),
						static_cast<unsigned>(N));
		}
		hipDeviceSynchronize();
		double exec_time = timer.get();
		cout << "    " << stride << "\t" << exec_time << "\t" << 20.0 * 32.0 * sizeof(NumericT) * N / exec_time * 1e-9 << endl;
	}
	return 0;
}
