#include <iostream>
#include <thread>

int main() {
	std::cout << "hardware concurrency: " << std::thread::hardware_concurrency() << std::endl;
	return 0;
}
