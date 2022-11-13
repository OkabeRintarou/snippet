cmake_minimum_required(VERSION 3.22)
project(libdrm_test)

set(CMAKE_CXX_STANDARD 20)

find_package(PkgConfig)

if (DEFINED ENV{DRM_DIR})
    set(DRM_DIR $ENV{DRM_DIR})
    set(DRM_INCLUDE_DIRS ${DRM_DIR}/include)
    link_libraries(${DRM_DIR}/lib64)
    set(DRM_LIBRARIES drm)
    set(DRM_AMDGPU_LIBRARIES drm_amdgpu)
else()
    pkg_check_modules(DRM REQUIRED libdrm)
    pkg_check_modules(DRM_AMDGPU REQUIRED libdrm_amdgpu)
    include_directories(${DRM_AMDGPU_INCLUDE_DIRS})
endif()

pkg_check_modules(GTEST REQUIRED gtest)
if (GTEST_FOUND)
    include_directories(${GTEST_INCLUDE_DIRS})
endif()

set(SRC_FILES
        amdgpu_util.cpp
        amdgpu_test.cpp
        amdgpu_device.cpp
        pm4_packet.cpp
        sdma_packet.cpp
        packet_common.cpp
        )

add_executable(vm_tests ${SRC_FILES} vm_tests.cpp)
target_link_libraries(vm_tests gtest drm drm_amdgpu)
add_executable(basic_tests ${SRC_FILES} basic_tests.cpp)
target_link_libraries(basic_tests gtest drm drm_amdgpu)
