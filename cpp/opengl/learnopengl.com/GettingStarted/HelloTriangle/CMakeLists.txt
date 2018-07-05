cmake_minimum_required(VERSION 3.10)
project(HelloTriangle)

set(CMAKE_CXX_STANDARD 11)

add_executable(hello_triangle hello_triangle.cc)
target_link_libraries(hello_triangle GL GLEW glfw)