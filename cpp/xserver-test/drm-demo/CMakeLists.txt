cmake_minimum_required(VERSION 3.16)
project(drm-demo)

find_package(PkgConfig)
pkg_check_modules(DRM REQUIRED libdrm)
include_directories(${DRM_INCLUDE_DIRS})

add_executable(single_buffer single_buffer.cc)
target_link_libraries(single_buffer drm)
add_executable(double_buffer double_buffer.cc)
target_link_libraries(double_buffer drm)
