#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <utility>
#include <boost/coroutine2/all.hpp>

using coro_t = boost::coroutines2::coroutine<std::string>;

struct FinalEOL {
	~FinalEOL() {
		std::cout << std::endl;
	}
};

int main() {
	const int num = 5, width = 15;
	coro_t::push_type writer(
		[&](coro_t::pull_type& in) {
		FinalEOL eol;

		// pull values from upstream, lay them out 'num' to a line
		for (;;) {
			for (int i = 0; i < num; i++) {
				// when we exhaust the input, stop
				if (!in) return;
				std::cout << std::setw(width) << in.get();
				// now that we've handled this item, advance to next
				in();
			}

			// after 'num' items, line break
			std::cout << std::endl;
		}
	});

	std::vector<std::string> words {
		"peas", "porrides", "hot", "peas", "porridge",
		"cold", "peas", "porridge", "in", "the",
		"pot", "nine", "days", "old",
	};

	using std::begin;
	using std::endl; // ADL name lookup
	std::copy(begin(words), end(words), begin(writer));
	std::cout << "\nDone";
	return 0;
}
