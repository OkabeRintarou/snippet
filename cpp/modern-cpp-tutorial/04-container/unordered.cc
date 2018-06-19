#include <iostream>
#include <map>
#include <string>
#include <unordered_map>

int main() {
  std::unordered_map<int, std::string> u = {
      {1, "1"},
      {2, "2"},
      {3, "3"},
  };

  std::map<int, std::string> v = {
      {1, "1"},
      {2, "2"},
      {3, "3"},
  };

  std::cout << "unordered_map:" << std::endl;
  for (const auto &x : u) {
    std::cout << x.first << "," << x.second << std::endl;
  }

  std::cout << "map:" << std::endl;
  for (const auto &x : v) {
    std::cout << x.first << "," << x.second << std::endl;
  }
  return 0;
}
