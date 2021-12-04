cmake_minimum_required(VERSION  3.15)

project(coro)

set(CMAKE_CXX_STANDARD 20)

add_executable(coro_echo_server echo_server.cpp)
target_link_libraries(coro_echo_server pthread)

add_executable(coro_echo_client echo_client.cpp)
target_link_libraries(coro_echo_client pthread)

