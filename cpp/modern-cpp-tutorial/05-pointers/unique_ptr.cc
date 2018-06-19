#include <iostream>
#include <memory>

int main() {
	std::unique_ptr<int> ptr = std::unique_ptr<int>(new int(10));
	// auto p2 = ptr; ERROR
	auto p2 = std::move(ptr);
	return 0;
}
