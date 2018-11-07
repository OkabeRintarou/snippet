#include "stackpartspec.hpp"
#include <iostream>

int main() {
  Stack<int *> ptrStack;
  ptrStack.push(new int{42});
  std::cout << *ptrStack.top() << std::endl;
  delete ptrStack.pop();
  return 0;
}
