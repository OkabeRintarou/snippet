#include <iostream>
#include <chrono>
#include <thread>

int main() {
	std::thread t([]() {
		for (;;) {
			std::cout << "Hello" << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(1));
		}
	});

	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	return 0;
}
