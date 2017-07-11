add_executable(simpe_echo echo/echo.cc echo/main.cc)
target_link_libraries(simpe_echo muduo_net muduo_base pthread)
