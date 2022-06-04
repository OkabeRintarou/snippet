add_library(Common STATIC
        Context.cc Texture2D.cc Object.cc gl_header.h
        Shader.cc Mesh.cc Model.cc Timer.cc)
target_link_libraries(Common GL GLEW glfw assimp)