#include <array>

constexpr fibonacci(const int n) {
	return n == 1 || n == 2 ? 1 : fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
	std::array<int,fibonacci(4)> arr;
	return 0;
}
