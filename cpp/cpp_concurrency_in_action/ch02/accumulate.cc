#include <iostream>
#include <thread>
#include <vector>
#include <numeric>
#include <algorithm>

template<typename I, typename T>
struct accumulate_block {
	void operator()(I first, I last, T &result) {
		result = std::accumulate(first, last, result);
	}
};

template<typename I, typename T>
T parallel_accumulate(I first, I last, T init) {
	auto length = std::distance(first, last);
	if (!length) {
		return init;
	}

	const auto min_per_thread = 25;
	const auto max_threads = (length + min_per_thread - 1) / min_per_thread;
	using thread_num_t = decltype(max_threads);
	const thread_num_t hardware_threads = std::thread::hardware_concurrency();
	const auto num_threads = std::min(
		(hardware_threads != 0) ? hardware_threads : thread_num_t(2), max_threads);

	const auto block_size = length / num_threads;

	std::vector<T> results(num_threads);
	std::vector<std::thread> threads(num_threads-1);
	I block_start = first;
	for (auto i = 0; i < num_threads - 1; i++) {
		I block_end = block_start;
		std::advance(block_end, block_size);

		threads[i] = std::thread(
			accumulate_block<I, T>(), block_start, block_end, std::ref(results[i]));
		block_start = block_end;
	}
	accumulate_block<I,T>()(block_start, last, results[num_threads - 1]);

	std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
	return std::accumulate(results.begin(), results.end(), init);
}

int main() {
	std::vector<int> nums(1000000);
	std::iota(nums.begin(), nums.end(), 0);
	using I = std::vector<int>::iterator;
	std::cout << parallel_accumulate<I, unsigned long long int>(nums.begin(), nums.end(), 0) << std::endl;
	return 0;
}
