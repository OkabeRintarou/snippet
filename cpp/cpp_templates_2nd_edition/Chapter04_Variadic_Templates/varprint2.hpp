#pragma once

#include <iostream>

template <typename T> void print(T arg) { std::cout << arg << '\n'; }

template <typename T, typename... Types> void print(T firstArg, Types... args) {
  std::cout << firstArg << '\n';
  print(args...);
}
