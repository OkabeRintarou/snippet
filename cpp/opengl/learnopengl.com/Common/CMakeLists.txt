add_library(Common STATIC Context.cc ShaderManager.cc)
target_link_libraries(Common GL GLEW glfw)