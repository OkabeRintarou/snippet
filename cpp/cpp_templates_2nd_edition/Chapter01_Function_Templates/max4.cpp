#include <bits/stdc++.h>

template<typename T>
T max(T a,T b) {
	std::cout << "max<T>()" << std::endl;
	return b < a ? a : b;
}

template<typename T>
T max(T a,T b,T c) {
	return max(max(a,b),c);
}

int max(int a,int c) {
	std::cout << "max(int,int)" << std::endl;
}

int main() {
	::max(47,11,33);
	return 0;
}
