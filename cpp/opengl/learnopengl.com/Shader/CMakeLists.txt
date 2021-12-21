cmake_minimum_required(VERSION 3.10)
project(Shader)

set(CMAKE_CXX_STANDARD 17)

add_executable(discolor discolor.cc)
target_link_libraries(discolor glfw3 GLEW GL X11 pthread Xrandr Xi dl)

add_executable(inout inout.cc)
target_link_libraries(inout glfw3 GLEW GL X11 pthread Xrandr Xi dl)

add_executable(Shader_Exercise1 Exercise1.cc)
target_link_libraries(Shader_Exercise1 glfw3 GLEW GL X11 pthread Xrandr Xi dl)

add_executable(Shader_Exercise2 Exercise2.cc)
target_link_libraries(Shader_Exercise2 glfw3 GLEW GL X11 pthread Xrandr Xi dl)

add_executable(Shader_Exercise3 Exercise3.cc)
target_link_libraries(Shader_Exercise3 glfw3 GLEW GL X11 pthread Xrandr Xi dl)
