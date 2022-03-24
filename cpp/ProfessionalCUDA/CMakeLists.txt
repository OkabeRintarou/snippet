cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 52)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

project(ProfessionCuda LANGUAGES CXX CUDA)

add_subdirectory(ch01)
add_subdirectory(ch02)