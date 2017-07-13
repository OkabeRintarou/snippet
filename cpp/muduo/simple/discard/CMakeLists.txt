add_executable(discard discard.cc main.cc)
target_link_libraries(discard muduo_base muduo_net pthread)
