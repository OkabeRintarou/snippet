cmake_minimum_required(VERSION 3.5)

add_executable(hello_test hello_test.cc hello.cc)
target_link_libraries(hello_test gtest pthread)

#add_test(HelloTest hello_test)
