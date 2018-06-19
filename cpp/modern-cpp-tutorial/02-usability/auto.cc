#include <iostream>


template<typename T,typename U>
auto add(T x,U y) {
	return x + y;
}

int main() {
{
	auto x = new auto(10);
	delete x;
}

{
	auto x = 1;
	auto y = 2;
	decltype(x+y) z;

	if (std::is_same<decltype(x),int>::value) {
		std::cout << "type x == int" << std::endl;
	}
	if (std::is_same<decltype(x),float>::value) {
		std::cout << "type x == float" << std::endl;
	}
	if (std::is_same<decltype(x),decltype(z)>::value) {
		std::cout << "type x == type z" << std::endl;
	}
{
	auto w = add<int,double>(1,2.0);
	if (std::is_same<decltype(w),double>::value) {
		std::cout << "w is double" << std::endl;
	}
	std::cout << w << std::endl;
}
}
	return 0;
}
