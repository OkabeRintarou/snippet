#include <cstdlib>
#include <ctime>
#include <cstdio>
#include "sum_ispc.h"

const int TOTAL_VALUES = 1024;
float a[TOTAL_VALUES];
float b[TOTAL_VALUES];
float c[TOTAL_VALUES];
float serial_c[TOTAL_VALUES];
static const float epsilon = 0.0001;

void init_data(float *c, int size) {
	for (int i = 0; i < size; i++) {
		c[i] = static_cast<float>(rand());
	}
}

static void serial_sum(int size, float *a,  float *b, float *c) {
	for (int i = 0; i < size; i++) {
		c[i] = a[i] + b[i];
	}
}

void verify_output(int size, float *ispc, float *serial) {
	for (int i = 0; i < size; i++) {
		if (ispc[i] - serial[i] > epsilon) {
			printf("diff %d: ispc<%.6f> serial<%.6f>\n", i, ispc[i], serial[i]);
			return;
		}
	}
	printf("====== Pass! ======\n");
}

int main() {
	init_data(a, TOTAL_VALUES);
	init_data(b, TOTAL_VALUES);

	ispc::sum(TOTAL_VALUES, a, b, c);
	serial_sum(TOTAL_VALUES, a, b, serial_c);
	verify_output(TOTAL_VALUES, c, serial_c);
	return 0;
}
