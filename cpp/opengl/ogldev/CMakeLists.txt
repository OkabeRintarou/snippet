cmake_minimum_required(VERSION 3.23)
project(ogldev)

set(CMAKE_CXX_STANDARD 20)

add_executable(tutorial01 tutorial01.cpp)
target_link_libraries(tutorial01 GL glut)
