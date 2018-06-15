#include <iostream>

struct Base {
virtual void foo(int) { std::cout << "Base::foo" << std::endl; }
};

struct SubClass:Base {
virtual void foo(int) override { std::cout << "SubClass::foo" << std::endl; }
};

int main() {
	Base *base = new SubClass();
	base->foo(0);
	return 0;
}
