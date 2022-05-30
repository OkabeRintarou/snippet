#include <stb_image.h>

#include "Context.h"
#include "Shader.h"
#include "Texture2D.h"
#include <cmath>
#include <iostream>

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
                                 "uniform mat4 transform;\n"
                                 "void main()\n"
                                 "{\n"
                                 "gl_Position = transform * vec4(aPos, 1.0);\n"
                                 "ourColor = aColor;\n"
                                 "TexCoord = aTexCoord;\n"
                                 "}\0";
const char *fragmentShaderSource =
    "#version 330 core\n"
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
  Shader shader(shader_source::from_string, vertexShaderSource, fragmentShaderSource);
  if (!shader.is_valid()) {
    std::cout << shader.message() << std::endl;
    return -1;
  }

  Texture2D texture1("../../Resources/Textures/container.jpg");
  Texture2D texture2("../../Resources/Textures/awesomeface.png");
  if (!texture1.is_valid()) {
    std::cerr << texture1.message() << std::endl;
    return -1;
  }
  if (!texture2.is_valid()) {
    std::cerr << texture2.message() << std::endl;
    return -1;
  }

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

  VertexArrayObjectBuilder<float> builder;
  auto vao = builder.stride(8).add(3).add(3).add(2).data(vertices, sizeof(vertices)).build();
  if (!vao) {
    std::cerr << "Fail to create vertex array object: " << vao.err_value() << std::endl;
    return -1;
  }

  auto &&VAO = vao.take_ok_value();

  GLuint EBO;
  glGenBuffers(1, &EBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);


  auto window = ctx->window();
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  glViewport(0, 0, width, height);

  shader.use_program();
  shader.set_int("texture1", 0);
  shader.set_int("texture2", 1);

  // render loop
  // -----------
  while (!glfwWindowShouldClose(window)) {

    processInput(window);

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glm::mat4 trans = glm::mat4(1.0f);
    trans = glm::translate(trans, glm::vec3(0.5f, -0.5f, 0.0f));
    trans =
        glm::rotate(trans, (float)glfwGetTime(), glm::vec3(0.0f, 0.0f, 1.0f));

    shader.set_mat4("transform", trans);

    // draw our first triangle
    shader.use_program();
    texture1.bind(0);
    texture2.bind(1);
    VAO.bind();
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    trans = glm::mat4(1.0f);
    trans = glm::scale(trans,
                       glm::vec3(std::abs(std::sin(glfwGetTime())),
                                 std::abs(std::cos(glfwGetTime())), 1.0f));
    trans = glm::translate(trans, glm::vec3(-0.5f, 0.5f, 0.0f));

    shader.set_mat4("transform", trans);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

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
