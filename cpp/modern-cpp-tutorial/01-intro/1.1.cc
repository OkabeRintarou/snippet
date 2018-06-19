#include "foo.h"
#include <iostream>
#include <functional>

int main() {
	[out=std::ref(std::cout << "Refult from C code: " << add(1,2))]() {
		out.get() << '\n';
	}();
	return 0;
}
