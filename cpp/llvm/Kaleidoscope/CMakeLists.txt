cmake_minimum_required(VERSION 3.5)
project(Kaleidoscope)

set(CMAKE_CXX_STANDARD 17)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "llvm library: ${LLVM_LIBRARIES}")
message(STATUS "llvm include dirs: ${LLVM_INCLUDE_DIRS}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs support core irreader)

set(SOURCE_FILES lexer.cpp parser.cpp codegen.cpp main.cpp)
add_executable(Kaleidoscope ${SOURCE_FILES})

target_link_libraries(Kaleidoscope ${llvm_libs})
