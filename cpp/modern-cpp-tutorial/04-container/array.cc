#include <array>
#include <iostream>

void foo(int *arr, int len) { return; }

int main() {
  constexpr int len = 4;
  std::array<int, 4> arr1 = {1, 2, 3, 4};
  std::array<int, len> arr2 = {1, 2, 3, 4};

  foo(&arr1[0], 4);
  foo(arr2.data(), 4);
  return 0;
}
