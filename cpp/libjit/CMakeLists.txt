add_executable(t1 t1.c)
target_link_libraries(t1 jit)
add_executable(t2 t2.c)
target_link_libraries(t2 jit)
add_executable(t3 t3.c)
target_link_libraries(t3 jit)

add_executable(t4 t4.cpp)
target_link_libraries(t4 jitplus jit)

add_executable(t5 t5.c)
target_link_libraries(t5 jit)
