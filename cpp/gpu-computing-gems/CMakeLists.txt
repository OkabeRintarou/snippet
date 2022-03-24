cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CUDA_ARCHITECTURES 52)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

project(gpu-computing-gems LANGUAGES CXX CUDA)

include_directories(common)

add_subdirectory(common)
add_subdirectory(ch04)