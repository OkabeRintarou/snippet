#include <functional>
#include <iostream>

using foo = void(int);

void functional(foo f) { f(1); }

int bar(int param) { return param * 2; }

void baz(int a, int b, int c) {
  std::cout << "baz: " << a << ", " << b << ", " << c << std::endl;
}

int main() {
  auto f = [](int value) { std::cout << value << std::endl; };
  functional(f);
  f(1);

  {
    std::function<int(int)> func = bar;

    int important = 10;
    std::function<int(int)> func2 = [&](int value) -> int {
      return 1 + value + important;
    };
    std::cout << func(10) << std::endl;
    std::cout << func2(10) << std::endl;
  }

  {
    auto bindFoo = std::bind(baz, std::placeholders::_1, 1, 2);
    bindFoo(1);
  }
  return 0;
}
