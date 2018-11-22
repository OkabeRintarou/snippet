#include "stacknontype.hpp"
#include <iostream>
#include <string>

int main() {
  Stack<int, 20> int20Stack;
  Stack<int, 40> int40Stack;
  Stack<std::string, 40> string40Stack;

  int20Stack.push(7);
  std::cout << int20Stack.top() << '\n';
  int20Stack.pop();

  string40Stack.push("hello");
  std::cout << string40Stack.top() << '\n';
  string40Stack.pop();

  return 0;
}
