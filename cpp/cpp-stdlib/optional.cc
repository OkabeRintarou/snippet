#include "test.h"
#include <complex>
#include <iostream>
#include <optional>
#include <vector>

void test_optional_creation() {
  // empty
  std::optional<int> oEmpty;
  std::optional<float> oFloat = std::nullopt;
  // direct
  std::optional<int> oInt(10);
  std::optional oIntDeduced(10); // deduction guides

  // make_optional
  auto oDouble = std::make_optional<std::complex<double>>(3.0, 4.0);

  // in_place
  std::optional<std::complex<double>> o7{std::in_place, 3.0, 4.0};

  // will call vector with direct init of {1, 2, 3}
  std::optional<std::vector<int>> oVec(std::in_place, {1, 2, 3});

  // copy from other optional
  auto oIntCopy = oInt;
}

void test_optional_in_place() {
  { std::optional<Test> u1(Test{}); }
  std::cout << "---" << std::endl;
  { std::optional<Test> u2{std::in_place}; }
}

int main() {
  test_optional_creation();
  test_optional_in_place();
  std::cout << sizeof(std::optional<double>) << std::endl;
  std::cout << sizeof(std::optional<int>) << std::endl;
  return 0;
}