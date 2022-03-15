export void sum(uniform int N, uniform float a[], uniform float b[], uniform float c[]) {
	for (int i = 0; i < N; i+=programCount) {
		c[programIndex+i] = a[programIndex+i] + b[programIndex+i];
	}
}
