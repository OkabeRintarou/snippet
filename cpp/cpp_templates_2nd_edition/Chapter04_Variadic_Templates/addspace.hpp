#pragma once

#include <iostream>

template <typename T> class AddSpace {
private:
  T const &ref;

public:
  AddSpace(T const &r) : ref(r) {}

  friend std::ostream &operator<<(std::ostream &os, AddSpace<T> s) {
    return os << s.ref << ' ';
  }
};

template <typename... T> void print(T const &... args) {
  (std::cout << ... << AddSpace(args)) << '\n';
}