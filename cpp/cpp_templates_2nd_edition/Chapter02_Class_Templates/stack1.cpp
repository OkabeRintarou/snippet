#include "stack1.hpp"
#include <iostream>
#include <string>
#include <utility>

int main() {
  Stack<int> intStack;
  Stack<std::string> stringStack;

  intStack.push(7);
  std::cout << intStack.top() << std::endl;

  stringStack.push("hello");
  std::cout << stringStack.top() << std::endl;
  stringStack.pop();
  std::cout << stringStack.empty() << std::endl;

  intStack.printOn(std::cout);
  std::cout << std::endl;

  return 0;
}
