#include <iostream>


struct Base {
virtual void foo() final { std::cout << "Base::foo" << std::endl; }
};

struct SubClass1 final:Base {

};

/* 
struct SubClass2:SubClass1 {
 // 非法:SubClass1 已 final
};
*/

/*
struct SubClass3:Base {
	void foo() {
		// 非法: foo 已final
	}
};
*/

int main() {
	return 0;
}

