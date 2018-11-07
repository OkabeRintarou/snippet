#include "stack3.hpp"
#include <deque>
#include <iostream>

template <typename T> using DequeStack = Stack<T, std::deque<T>>;

int main() {
  Stack<int> intStack;
  for (int i = 0; i < 10; i++) {
    intStack.push(i);
  }
  while (!intStack.empty()) {
    std::cout << intStack.top() << std::endl;
    intStack.pop();
  }

  Stack<double, std::deque<double>> doubleStack;
  doubleStack.push(42.42);
  std::cout << doubleStack.top() << std::endl;
  doubleStack.pop();

  DequeStack<int> intDequeStack;
  intDequeStack.push(10);
  std::cout << intDequeStack.top() << std::endl;
  return 0;
}
