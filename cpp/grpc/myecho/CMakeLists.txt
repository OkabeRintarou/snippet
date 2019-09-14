cmake_minimum_required(VERSION 3.14)
project(myecho)

add_executable(myserver server.cpp myrpc.cpp echo.pb.cc rpc_meta.pb.cc)
target_link_libraries(myserver brpc protobuf gflags boost_system)

add_executable(myclient client.cpp myrpc.cpp echo.pb.cc rpc_meta.pb.cc)
target_link_libraries(myclient brpc protobuf gflags boost_system)

