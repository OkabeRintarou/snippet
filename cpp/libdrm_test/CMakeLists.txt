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
    main.cpp)

add_executable(libdrm_test ${SRC_FILES})
target_link_libraries(libdrm_test ${GTEST_LIBRARIES} ${DRM_LDFLAGS} ${DRM_AMDGPU_LDFLAGS} pthread)
