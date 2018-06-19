#include <iostream>

void learn_lambda_func_1() {
  int value_1 = 1;
  auto copy_value_1 = [value_1] { // 被捕获的变量在lambda创建时拷贝
    return value_1;
  };
  value_1 = 100;
  auto stored_value_1 = copy_value_1(); // stored_value_1 == 1,value_1 == 100
  std::cout << "value_1: " << value_1 << std::endl;
  std::cout << "stored_value_1: " << stored_value_1 << std::endl;
}

void learn_lambda_func_2() {
  int value_2 = 2;
  auto copy_value_2 = [&value_2] { return value_2; };
  value_2 = 100;
  auto stored_value_2 = copy_value_2(); // stored_value_2 == 100,value_2 == 100

  std::cout << "value_2: " << value_2 << std::endl;
  std::cout << "stored_value_2: " << stored_value_2 << std::endl;
}

int main() {
  learn_lambda_func_1();
  learn_lambda_func_2();
  return 0;
}
