add_library(Common STATIC Context.cc ShaderManager.cc Camera.cc Texture2D.cc Object.cc gl_header.h Shader.h Shader.cc)
target_link_libraries(Common GL GLEW glfw)