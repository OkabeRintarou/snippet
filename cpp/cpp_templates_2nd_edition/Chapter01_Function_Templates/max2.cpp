#include <bits/stdc++.h>

int max(int a,int b) {
	std::cout << "nontemplate" << std::endl;
	return b < a ? a : b;
}

template<typename T>
T max(T a,T b) {
	std::cout << "template" << std::endl;
	return b < a ? a : b;
}

int main() {
	::max(7,42);
	::max(7.0,42.0);
	::max('a','b');
	::max<>(7,42);
	::max<double>(7,42);
	// ::max<>('a',42.7);	ERROR
	return 0;
}
