#include <iostream>
#include <string>

std::string operator"" _wow1(const char *wow1, size_t len) {
  return std::string(wow1) + "woooooooow, amazing";
}

std::string operator"" _wow2(unsigned long long i) {
  return std::to_string(i) + "woooooooow, amazing";
}

int main() {
  std::string str = R"(C:\\What\\The\\Fxxk)";
  std::cout << str << std::endl;

  {
    auto str = "abc"_wow1;
    auto num = 1_wow2;
    std::cout << str << std::endl;
    std::cout << num << std::endl;
  }
  return 0;
}
