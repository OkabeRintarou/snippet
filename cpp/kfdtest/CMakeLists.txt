cmake_minimum_required(VERSION 3.22)
project(kfdtest)

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

if (DEFINED ENV{LIBHSAKMT_PATH})
    set(LIBHSAKMT_PATH $ENV{LIBHSAKMT_PATH})
    message("LIBHSAKMT_PATH environment variable is set")
else()
    if (${ROCM_INSTALL_PATH})
        set (ENV{PKG_CONFIG_PATH} ${ROCM_INSTALL_PATH}/share/pkgconfig)
    else()
        set (ENV{PKG_CONFIG_PATH /opt/rocm/share/pkgconfig)
    endif()

    pkg_check_modules(HSAKMT libhsakmt)

    if (NOT HSAKMT_FOUND)
        set (LIBHSAKMT_PATH $ENV{OUT_DIR})
    endif()
endif()

if (DEFINED LIBHSAKMT_PATH)
    set(HSAKMT_LIBRARIES_DIRS ${LIBHSAKMT_PATH}/lib)
    set(HSAKMT_LIBRARIES hsakmt)
endif ()

message("Find libhsakmt at ${HSAKMT_LIBRARY_DIRS}")

pkg_check_modules(GTEST REQUIRED gtest)
if (GTEST_FOUND)
    include_directories(${GTEST_INCLUDE_DIRS})
endif()

set(SRC_FILES
        KFDTestMain.cpp
        KFDBaseComponentTest.cpp
        )

add_executable(kfdtest ${SRC_FILES})
target_link_libraries(kfdtest ${HSAKMT_LIBRARIES} ${GTEST_LIBRARIES} ${DRM_LDFLAGS} ${DRM_AMDGPU_LDFLAGS} pthread m numa)
