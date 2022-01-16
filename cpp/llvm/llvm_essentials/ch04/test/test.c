#include <stdio.h>

int fib(unsigned a) {
	if (a <= 2) {
		return 1;
	}
	return fib(a - 1) + fib(a - 2);
}

void print() {
	printf("%d\n", fib(10));
}

int main() {
	print();
	return 0;
}

