cmake_minimum_required(VERSION  3.15)

project(tcp)

add_executable(tcp_client client.cpp)
target_link_libraries(tcp_client boost_thread pthread)

add_executable(tcp_server server.cpp)
target_link_libraries(tcp_server boost_thread pthread)

add_executable(async_tcp_server async_server.cpp)
target_link_libraries(async_tcp_server boost_thread pthread)