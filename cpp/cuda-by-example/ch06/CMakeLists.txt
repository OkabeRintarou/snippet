add_executable(ray_nonconst ray_nonconst.cu)
target_link_libraries(ray_nonconst GL glut)
add_executable(ray_const ray_const.cu)
target_link_libraries(ray_const GL glut)