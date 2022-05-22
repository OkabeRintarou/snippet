#include "test.h"
#include <iostream>

Test::Test() { std::cout << "Test::Test()" << std::endl; }

Test::~Test() { std::cout << "Test::~Test()" << std::endl; }

Test::Test(const Test &) {
  std::cout << "Test::Test(const Test&)" << std::endl;
}

Test &Test::operator=(const Test &) {
  std::cout << "Test& Test::operator=(const Test&)" << std::endl;
  return *this;
}

Test::Test(Test &&) { std::cout << "Test::Test(Test&&)" << std::endl; }

Test &Test::operator=(Test &&) {
  std::cout << "Test& Test::operator=(Test&&)" << std::endl;
  return *this;
}