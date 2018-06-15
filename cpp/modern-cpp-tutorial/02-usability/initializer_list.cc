#include <initializer_list>
#include <vector>
#include <iostream>

class MagicFoo {
public:
	std::vector<int> vec;
	MagicFoo(std::initializer_list<int> list) {
		for (auto it = list.begin(); it != list.end(); ++it) {
			vec.push_back(*it);
		}
	}

	void foo(std::initializer_list<int> list) {
		for (std::initializer_list<int>::iterator it = list.begin();
				it != list.end(); ++it) {
			vec.push_back(*it);
		}
	}
};

int main() {
	MagicFoo magicFoo = {1,2,3,4,5};
	magicFoo.foo({6,7,8,9});
	std::cout << "magicFoo: ";
	for (std::vector<int>::iterator it = magicFoo.vec.begin(); it != magicFoo.vec.end(); ++it) {
		std::cout << *it << " ";
	}
	return 0;
}
