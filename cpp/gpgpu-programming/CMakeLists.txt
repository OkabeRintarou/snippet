cmake_minimum_required(VERSION 3.15)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CUDA_STANDARD 17)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_ARCHITECTURES 52)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

project(gpgpu LANGUAGES CXX CUDA)

include_directories(common)

add_subdirectory(ch01)
add_subdirectory(ch02)
add_subdirectory(ch04)
add_subdirectory(ch05)
add_subdirectory(common)