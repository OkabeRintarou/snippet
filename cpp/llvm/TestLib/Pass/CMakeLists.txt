cmake_minimum_required(VERSION 3.14.3)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "")

find_package(LLVM REQUIRED CONFIG)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

list(APPEND CMAKE_PREFIX_PATH "${LT_LLVM_INSTALL_DIR}/lib/cmake/llvm")
include(AddLLVM)

if (NOT ${LLVM_ENABLE_RTTI})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

add_subdirectory(PassManager)
add_subdirectory(LegacyPassManager)

add_llvm_pass_plugin(OpcodeCounter OpcodeCounter.cpp)