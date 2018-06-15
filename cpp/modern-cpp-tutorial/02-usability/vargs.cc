#include <iostream>
#include <string>

template<typename T>
void print(T value) {
	std::cout << value << std::endl;
}

template<typename T,typename... Args>
void print(T value,Args... args) {
	std::cout << value << std::endl;
	print(args...);
}

template<typename... Args>
void magic(Args... args) {
	std::cout << sizeof...(args) << std::endl;
}

template<typename T,typename ...Args>
void print2(T value,Args... args) {
	std::cout << value << std::endl;
	if constexpr (sizeof... (args) > 0) print2(args...);
}

int main() {
	print(1,2,"hello");
	print2(1,2,"hello");
	return 0;
}

