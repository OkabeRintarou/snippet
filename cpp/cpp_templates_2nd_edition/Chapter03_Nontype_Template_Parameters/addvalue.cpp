#include "addvalue.hpp"
#include <algorithm>
#include <iostream>

int main() {
  int a[] = {1, 2, 3, 4};
  const int length = sizeof(a) / sizeof(*a);
  int b[length];
  std::transform(a, a + length, b, addValue<5, int>);
  std::for_each(b, b + length,
                [](int const &value) { std::cout << value << std::endl; });
  return 0;
}
