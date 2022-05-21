#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "Context.h"
#include "ShaderManager.h"
#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

void key_callback(GLFWwindow *window, int key, int scancode, int action,
                  int mode);
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

const char *vertexShaderSource = "#version 330 core\n"
                                 "layout (location = 0) in vec3 aPos;\n"
                                 "layout (location = 1) in vec3 aColor;\n"
                                 "layout (location = 2) in vec2 aTexCoord;\n"
                                 "out vec3 ourColor;\n"
                                 "out vec2 TexCoord;\n"
                                 "void main()\n"
                                 "{\n"
                                 "gl_Position = vec4(aPos, 1.0);\n"
                                 "ourColor = aColor;\n"
                                 "TexCoord = aTexCoord;\n"
                                 "}\0";
const char *fragmentShaderSource = "#version 330 core\n"
                                   "out vec4 color;\n"
                                   "in vec3 ourColor;\n"
                                   "in vec2 TexCoord;\n"
                                   "uniform sampler2D texture1;\n"
                                   "uniform sampler2D texture2;\n"
                                   "void main()\n"
                                   "{\n"
                                   "color=mix(texture(texture1, TexCoord), "
                                   "texture(texture2, vec2(1.0 - TexCoord.x, TexCoord.y)), 0.2);\n"
                                   "}\0";

int main() {

  stbi_set_flip_vertically_on_load(true);

  auto ctx = Context::init(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL");
  if (!ctx) {
    std::cerr << "Fail to init context" << std::endl;
    return -1;
  }
  // compile and link shader
  ShaderManager &sm = ctx->shader_manager();
  if (auto shader = sm.load(shader_source::from_string, shader_type::vertex,
                            vertexShaderSource);
      !shader) {
    std::cerr << "Fail to load vertex shader: " << shader.err_value()
              << std::endl;
    return -1;
  }
  if (auto shader = sm.load(shader_source::from_string, shader_type::fragment,
                            fragmentShaderSource);
      !shader) {
    std::cerr << "Fail to load fragment shader: " << shader.err_value()
              << std::endl;
    return -1;
  }
  auto pgr = sm.link();
  if (!pgr) {
    std::cout << "Fail to link program shader: " << pgr.err_value()
              << std::endl;
    return -1;
  }
  GLuint shader_program = pgr.ok_value();

  unsigned texture1, texture2;

  // texture1
  // --------
  glGenTextures(1, &texture1);
  glBindTexture(GL_TEXTURE_2D, texture1);
  // set the texture1 wrapping/filtering options(on the currently bound texture1
  // object)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  int img_width, img_height, img_channels;
  unsigned char *img_data =
      stbi_load("../../Resources/Textures/container.jpg", &img_width,
                &img_height, &img_channels, 0);
  if (!img_data) {
    std::cerr << "Fail to load image container.jpg" << std::endl;
    return -1;
  }
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_width, img_height, 0, GL_RGB,
               GL_UNSIGNED_BYTE, img_data);
  glGenerateMipmap(GL_TEXTURE_2D);
  stbi_image_free(img_data);

  // texture2
  // --------
  glGenTextures(1, &texture2);
  glBindTexture(GL_TEXTURE_2D, texture2);
  // set the texture wrapping parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  // set texture filtering parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  img_data = stbi_load("../../Resources/Textures/awesomeface.png", &img_width,
                       &img_height, &img_channels, 0);
  if (!img_data) {
    std::cerr << "Fail to load image awesomeface.png" << std::endl;
    return -1;
  }
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_width, img_height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, img_data);
  glGenerateMipmap(GL_TEXTURE_2D);
  stbi_image_free(img_data);

  // set up vertex data (and buffer(s)) and configure vertex attributes
  // ------------------------------------------------------------------
  float vertices[] = {
      // position:colors:texture1 coords
      0.5f,  0.5f,  0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, // top right
      0.5f,  -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, // bottom right
      -0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, // bottom left
      -0.5f, 0.5f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, // top left
  };
  unsigned int indices[] = {
      0, 1, 3, // first triangle
      1, 2, 3  // second triangle
  };

  GLuint VBO, VAO, EBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);

  // bind the Vertex Array Object first, then bind and set vertex buffer(s), can
  // then configure vertex attribute(s)
  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);

  // position attribute
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), nullptr);
  glEnableVertexAttribArray(0);
  // color attribute
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                        (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);
  // texture1 coord attribute
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                        (void *)(6 * sizeof(float)));
  glEnableVertexAttribArray(2);

  auto window = ctx->window();
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  glViewport(0, 0, width, height);

  glUseProgram(shader_program);
  glUniform1i(glGetUniformLocation(shader_program, "texture1"), 0);
  glUniform1i(glGetUniformLocation(shader_program, "texture2"), 1);
  // render loop
  // -----------
  while (!glfwWindowShouldClose(window)) {

    processInput(window);

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, texture1);
    // draw our first triangle
    glUseProgram(shader_program);
    glUseProgram(shader_program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture2);

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);

  glfwTerminate();

  return 0;
}

void key_callback(GLFWwindow *window, int key, int scancode, int action,
                  int mode) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GL_TRUE);
  }
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {}

void processInput(GLFWwindow *window) {}
