cmake_minimum_required(VERSION 3.10)
project(Shader)

set(CMAKE_CXX_STANDARD 11)

add_executable(discolor discolor.cc)
target_link_libraries(discolor GL GLEW glfw)

add_executable(inout inout.cc)
target_link_libraries(inout GL GLEW glfw)

add_executable(attributes attributes.cc)
target_link_libraries(attributes GL GLEW glfw)

add_executable(Shader_Exercise1 Exercise1.cc)
target_link_libraries(Shader_Exercise1 GL GLEW glfw)

add_executable(Shader_Exercise2 Exercise2.cc)
target_link_libraries(Shader_Exercise2 GL GLEW glfw)

add_executable(Shader_Exercise3 Exercise3.cc)
target_link_libraries(Shader_Exercise3 GL GLEW glfw)