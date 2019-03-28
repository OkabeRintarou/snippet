#include <iostream>
#include <thread>

void do_something(int i) {
	std::cout << "do something" << std::endl;
}

struct func {
	int &i;

	func(int &i_) : i(i_) {}

	void operator()() {
		for (auto j = 0; j < 10; j++) {
			do_something(i);
		}
	}
};

class thread_guard {
	std::thread &t;
public:
	explicit thread_guard(std::thread &t_) : t(t_) {}
	~thread_guard() {
		if (t.joinable()) {
			t.join();
			std::cout << "thread[" << t.get_id() << "] exiting..." << std::endl;
		}
	}

	thread_guard(thread_guard const &) = delete;
	thread_guard &operator=(thread_guard const &) = delete;
};

int main() {
	int some_local_state = 0;
	func my_func(some_local_state);
	std::thread t(my_func);
	thread_guard g(t);

	auto do_something_in_current_thread = []() {
		throw "invalid argument";
	};
	try {
		do_something_in_current_thread();
	} catch (const char *e) {
		std::cout << "current thread throw exception: " << e << std::endl;
	}
	return 0;
}
