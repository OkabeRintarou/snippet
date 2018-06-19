#include <iostream>
#include <tuple>

auto gen_student(int id) {
  if (id == 0) {
    return std::make_tuple(3.8, 'A', "张三");
  } else if (id == 1) {
    return std::make_tuple(2.9, 'C', "李四");
  } else if (id == 2) {
    return std::make_tuple(1.7, 'D', "王五");
  }
  return std::make_tuple(0.0, 'D', "null");
}

template <typename T> auto tuple_len(T &tple) {
  return std::tuple_size<T>::value;
}

int main() {
  auto student = gen_student(0);
  std::cout << "ID 0, "
            << "GPA: " << std::get<0>(student) << ", "
            << "成绩: " << std::get<1>(student) << ", "
            << "姓名: " << std::get<2>(student) << std::endl;

  double gpa;
  char grade;
  std::string name;

  std::tie(gpa, grade, name) = student;
  std::cout << "ID 0, "
            << "GPA: " << std::get<double>(student) << ", "
            << "成绩: " << std::get<char>(student) << ", "
            << "姓名: " << std::get<const char *>(student) << std::endl;

  std::cout << tuple_len(student) << std::endl;
  return 0;
}
