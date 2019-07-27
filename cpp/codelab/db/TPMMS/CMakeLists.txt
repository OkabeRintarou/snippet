cmake_minimum_required(VERSION 3.14)
project(TPMMS)

set(CMAKE_CXX_STANDARD 14)


add_executable(gen_record gen_record.cpp record.cpp)
target_link_libraries(gen_record gflags)

SET(GCC_COMPILE_FLAGS "-O2")
add_executable(sort sort.cpp record.cpp timer.cpp)
target_link_libraries(sort gflags glog)
add_definitions(${GCC_COMPILE_FLAGS})