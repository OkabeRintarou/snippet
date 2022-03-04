cmake_minimum_required(VERSION 3.14)
project(amdgpu_test)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

include(googletest)

include_directories(/usr/include/libdrm)
add_executable(basic_test basic_test.cc amdgpu_device.cc amdgpu_bo.cc amdgpu_context.cc)
target_link_libraries(basic_test gtest_main drm drm_amdgpu)

add_executable(bo_test bo_test.cc amdgpu_device.cc amdgpu_bo.cc amdgpu_context.cc)
target_link_libraries(bo_test gtest_main drm drm_amdgpu)

add_executable(result_test result_test.cc)
target_link_libraries(result_test gtest_main)

include(GoogleTest)
gtest_discover_tests(basic_test)
gtest_discover_tests(bo_test)
gtest_discover_tests(result_test)
