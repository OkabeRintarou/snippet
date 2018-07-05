cmake_minimum_required(VERSION 3.10)
project(HelloTriangle)

set(CMAKE_CXX_STANDARD 11)

add_executable(hello_triangle hello_triangle.cc )
target_link_libraries(hello_triangle GL GLEW glfw)

add_executable(Exercise1 Exercise1.cc)
target_link_libraries(Exercise1 GL GLEW glfw)

add_executable(Exercise2 Exercise2.cc)
target_link_libraries(Exercise2 GL GLEW glfw)

add_executable(Exercise3 Exercise3.cc)
target_link_libraries(Exercise3 GL GLEW glfw)