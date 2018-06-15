#include <iostream>

template<typename ...T>
auto average(T ...t) {
	return (t + ...) / sizeof... (t);
}

int main() {
	std::cout << average(1,2,3,4,5,6) << std::endl;
	return 0;
}
