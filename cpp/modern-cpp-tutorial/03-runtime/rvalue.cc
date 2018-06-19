#include <iostream>
#include <string>
#include <utility>
#include <vector>

void reference(std::string &str) { std::cout << "lvalue" << std::endl; }

void reference(std::string &&str) { std::cout << "rvalue" << std::endl; }

class A {
public:
  int *pointer;
  A() : pointer(new int(1)) { std::cout << "constructor" << std::endl; }
  A(A &a) : pointer(new int(*a.pointer)) {
    std::cout << "copy constructor" << std::endl;
  }
  A(A &&a) : pointer(a.pointer) {
    a.pointer = nullptr;
    std::cout << "move constructor" << std::endl;
  }
  ~A() { std::cout << "deconstructor" << std::endl; }
};

A return_rvalue(bool test) {
  A a, b;
  if (test) {
    return a;
  }
  return b;
}

int main() {
  {
    std::string lv1 = "string";
    std::string &&rv1 = std::move(lv1);
    reference(rv1); // 输出rvalue,rv1引用的是右值,但本身是一个左值
    reference(lv1);
  }

  {
    A obj = return_rvalue(false);
    std::cout << "obj: " << std::endl;
    std::cout << obj.pointer << std::endl;
    std::cout << *obj.pointer << std::endl;
  }

  {
    std::string str = "hello,world";
    std::vector<std::string> vec;
    vec.push_back(str);
    std::cout << "str: " << str << std::endl;

    vec.push_back(std::move(str));
    std::cout << "str: " << str << std::endl;
  }
  return 0;
}
