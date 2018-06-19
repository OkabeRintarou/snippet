#include <iostream>
#include <memory>

void foo(std::shared_ptr<int> i) {
	(*i)++;
}

int main() {
{
	auto pointer = std::make_shared<int>(10);
	foo(pointer);
	std::cout << *pointer << std::endl;
}

{
	auto pointer = std::make_shared<int>(10);
	auto pointer2 = pointer;
	auto pointer3 = pointer;

	std::cout << pointer.use_count() << std::endl;
}
	return 0;
}
