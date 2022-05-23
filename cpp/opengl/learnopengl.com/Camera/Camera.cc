
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "Context.h"
#include "ShaderManager.h"
#include <cmath>
#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

void key_callback(GLFWwindow *window, int key, int scancode, int action,
                  int mode);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
glm::vec3 camera_pos = glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 camera_front = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f);
float delta_time = 0.0f; // time between current frame and last frame
float last_frame = 0.0f; // time of last frame
float last_x = 400.0f, last_y = 300.0f;
float pitch = 0.0f, yaw = -90.0f;
bool first_mouse = true;
float fov = 45.0f;

const char *vertexShaderSource =
    "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec2 aTexCoord;\n"
    "out vec2 TexCoord;\n"
    "uniform mat4 model;\n"
    "uniform mat4 view;\n"
    "uniform mat4 projection;\n"
    "void main()\n"
    "{\n"
    "gl_Position = projection * view * model * vec4(aPos, 1.0);\n"
    "TexCoord = aTexCoord;\n"
    "}\0";
const char *fragmentShaderSource =
    "#version 330 core\n"
    "out vec4 color;\n"
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

  glfwSetInputMode(ctx->window(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwSetCursorPosCallback(ctx->window(), mouse_callback);
  glfwSetScrollCallback(ctx->window(), scroll_callback);

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
      -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, 0.5f,  -0.5f, -0.5f, 1.0f, 0.0f,
      0.5f,  0.5f,  -0.5f, 1.0f, 1.0f, 0.5f,  0.5f,  -0.5f, 1.0f, 1.0f,
      -0.5f, 0.5f,  -0.5f, 0.0f, 1.0f, -0.5f, -0.5f, -0.5f, 0.0f, 0.0f,

      -0.5f, -0.5f, 0.5f,  0.0f, 0.0f, 0.5f,  -0.5f, 0.5f,  1.0f, 0.0f,
      0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
      -0.5f, 0.5f,  0.5f,  0.0f, 1.0f, -0.5f, -0.5f, 0.5f,  0.0f, 0.0f,

      -0.5f, 0.5f,  0.5f,  1.0f, 0.0f, -0.5f, 0.5f,  -0.5f, 1.0f, 1.0f,
      -0.5f, -0.5f, -0.5f, 0.0f, 1.0f, -0.5f, -0.5f, -0.5f, 0.0f, 1.0f,
      -0.5f, -0.5f, 0.5f,  0.0f, 0.0f, -0.5f, 0.5f,  0.5f,  1.0f, 0.0f,

      0.5f,  0.5f,  0.5f,  1.0f, 0.0f, 0.5f,  0.5f,  -0.5f, 1.0f, 1.0f,
      0.5f,  -0.5f, -0.5f, 0.0f, 1.0f, 0.5f,  -0.5f, -0.5f, 0.0f, 1.0f,
      0.5f,  -0.5f, 0.5f,  0.0f, 0.0f, 0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

      -0.5f, -0.5f, -0.5f, 0.0f, 1.0f, 0.5f,  -0.5f, -0.5f, 1.0f, 1.0f,
      0.5f,  -0.5f, 0.5f,  1.0f, 0.0f, 0.5f,  -0.5f, 0.5f,  1.0f, 0.0f,
      -0.5f, -0.5f, 0.5f,  0.0f, 0.0f, -0.5f, -0.5f, -0.5f, 0.0f, 1.0f,

      -0.5f, 0.5f,  -0.5f, 0.0f, 1.0f, 0.5f,  0.5f,  -0.5f, 1.0f, 1.0f,
      0.5f,  0.5f,  0.5f,  1.0f, 0.0f, 0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
      -0.5f, 0.5f,  0.5f,  0.0f, 0.0f, -0.5f, 0.5f,  -0.5f, 0.0f, 1.0f};

  GLuint VBO, VAO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);

  // bind the Vertex Array Object first, then bind and set vertex buffer(s), can
  // then configure vertex attribute(s)
  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  // position attribute
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
  glEnableVertexAttribArray(0);
  // texture1 coord attribute
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                        (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  auto window = ctx->window();
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  glViewport(0, 0, width, height);

  glUseProgram(shader_program);
  glUniform1i(glGetUniformLocation(shader_program, "texture1"), 0);
  glUniform1i(glGetUniformLocation(shader_program, "texture2"), 1);

  glm::vec3 cube_positions[] = {
      glm::vec3(0.0f, 0.0f, 0.0f),    glm::vec3(2.0f, 5.0f, -15.0f),
      glm::vec3(-1.5f, -2.2f, -2.5f), glm::vec3(-3.8f, -2.0f, -12.3f),
      glm::vec3(2.4f, -0.4f, -3.5f),  glm::vec3(-1.7f, 3.0f, -7.5f),
      glm::vec3(1.3f, -2.0f, -2.5f),  glm::vec3(1.5f, 2.0f, -2.5f),
      glm::vec3(1.5f, 0.2f, -1.5f),   glm::vec3(-1.3f, 1.0f, -1.5f)};

  unsigned model_loc = glGetUniformLocation(shader_program, "model");
  unsigned view_loc = glGetUniformLocation(shader_program, "view");
  unsigned projection_loc = glGetUniformLocation(shader_program, "projection");

  glEnable(GL_DEPTH_TEST);
  // render loop
  // -----------
  while (!glfwWindowShouldClose(window)) {
    auto current_time = static_cast<float>(glfwGetTime());
    delta_time = current_time - last_frame;
    last_frame = current_time;

    processInput(window);

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glm::mat4 projection = glm::perspective(
        glm::radians(fov), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
    glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm::value_ptr(projection));
    glm::mat4 view =
        glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm::value_ptr(view));

    glBindTexture(GL_TEXTURE_2D, texture1);
    // draw our first triangle
    glUseProgram(shader_program);
    glUseProgram(shader_program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture2);

    glBindVertexArray(VAO);

    for (unsigned i = 0; i < 10; i++) {
      glm::mat4 model = glm::mat4(1.0f);
      model = glm::translate(model, cube_positions[i]);
      float angle = 20.0f * i;
      if ((i % 3) == 0) {
        angle = glfwGetTime() * 25.0f;
      }
      model =
          glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
      glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm::value_ptr(model));

      glDrawArrays(GL_TRIANGLES, 0, 36);
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);

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

void processInput(GLFWwindow *window) {
  const float camera_speed = 2.5f * delta_time;
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    camera_pos += camera_speed * camera_front;
  } else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    camera_pos -= camera_speed * camera_front;
  } else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
    camera_pos -=
        glm::normalize(glm::cross(camera_front, camera_up)) * camera_speed;
  } else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
    camera_pos +=
        glm::normalize(glm::cross(camera_front, camera_up)) * camera_speed;
  }
}

void mouse_callback(GLFWwindow *window, double xpos, double ypos) {
  if (first_mouse) {
    last_x = xpos;
    last_y = xpos;
    first_mouse = false;
  }
  float xoffset = xpos - last_x;
  float yoffset = ypos - last_y;
  last_x = xpos;
  last_y = ypos;

  const float sensitivity = 0.1f;
  xoffset *= sensitivity;
  yoffset *= sensitivity;

  yaw += xoffset;
  pitch += yoffset;

  if (pitch > 89.f) {
    pitch = 89.0f;
  } else if (pitch < -89.0f) {
    pitch = -89.0f;
  }

  glm::vec3 direction;
  direction.x = cosf(glm::radians(yaw)) * cosf(glm::radians(pitch));
  direction.y = sinf(glm::radians(pitch));
  direction.z = sinf(glm::radians(yaw)) * cosf(glm::radians(pitch));
  camera_front = glm::normalize(direction);
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
  fov -= (float)yoffset;
  if (fov < 1.0f) {
    fov = 1.0f;
  } else if (fov > 65.0f) {
    fov = 65.0f;
  }
}