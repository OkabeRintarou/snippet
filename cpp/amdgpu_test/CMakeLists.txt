cmake_minimum_required(VERSION 3.14)
project(amdgpu_test)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

include(googletest)

add_executable(basic_test basic_test.cc)
target_link_libraries(basic_test gtest_main)

include(GoogleTest)
gtest_discover_tests(basic_test)
