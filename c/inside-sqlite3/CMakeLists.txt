cmake_minimum_required(VERSION 3.12)
project(inside_sqlite3 C)

set(CMAKE_C_STANDARD 99)

add_executable(basic open.c)
target_link_libraries(basic sqlite3)

add_executable(cmd cmd.c)
target_link_libraries(cmd sqlite3)

add_executable(thread1 thread1.c)
target_link_libraries(thread1 pthread sqlite3)

add_executable(multidb multidb.c)
target_link_libraries(multidb sqlite3)