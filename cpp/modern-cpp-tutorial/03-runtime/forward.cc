#include <iostream>
#include <utility>

void reference(int &v) { std::cout << "lvalue reference" << std::endl; }

void reference(int &&v) { std::cout << "rvalue reference" << std::endl; }

template <typename T> void pass(T &&v) {
  std::cout << "normal:";
  reference(v);
  std::cout << "std::move:";
  reference(std::move(v));
  std::cout << "std::forward:";
  reference(std::forward<T>(v));
}

int main() {
  std::cout << "rvalue parameter:" << std::endl;
  pass(1);

  std::cout << "lvalue parameter:" << std::endl;
  int v = 1;
  pass(v);
  return 0;
}
