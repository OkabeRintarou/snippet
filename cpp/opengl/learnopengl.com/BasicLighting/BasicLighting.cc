
#include <stb_image.h>

#include "Camera.h"
#include "Context.h"
#include "Shader.h"
#include "Texture2D.h"
#include <cmath>
#include <iostream>

void key_callback(GLFWwindow *window, int key, int scancode, int action,
                  int mode);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float last_x = SCR_WIDTH / 2.0f, last_y = SCR_HEIGHT / 2.0f;
bool first_mouse = true;

// timing
float delta_time = 0.0f; // time between current frame and last frame
float last_frame = 0.0f; // time of last frame

// lighting
glm::vec3 light_pos(1.2f, 1.0f, 2.0f);


int main() {

  stbi_set_flip_vertically_on_load(true);

  auto ctx = Context::init(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL");
  if (!ctx) {
    std::cerr << "Fail to init context" << std::endl;
    return -1;
  }

  // glfwSetInputMode(ctx->window(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwSetCursorPosCallback(ctx->window(), mouse_callback);
  glfwSetScrollCallback(ctx->window(), scroll_callback);

  // compile and link shader

  Shader shader(shader_source::from_file,
                "../../Resources/Shader/BasicLighting_Vertex.glsl",
                "../../Resources/Shader/BasicLighting_Frag.glsl");
  if (!shader.is_valid()) {
    std::cerr << "Fail to create shader: " << shader.message() << std::endl;
    return -1;
  }

  Shader light_shader(shader_source::from_file,
                      "../../Resources/Shader/BasicLighting_Vertex.glsl",
                      "../../Resources/Shader/BasicLighting_Light_Frag.glsl");
  if (!light_shader.is_valid()) {
    std::cerr << "Fail to create light shader: " << light_shader.message()
              << std::endl;
    return -1;
  }

  // set up vertex data (and buffer(s)) and configure vertex attributes
  // ------------------------------------------------------------------
  float vertices[] = {
      -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
      0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
      0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
      0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
      -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
      -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,

      -0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
      0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
      0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
      0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
      -0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
      -0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,

      -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
      -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
      -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
      -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
      -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
      -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,

      0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
      0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
      0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
      0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
      0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
      0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,

      -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
      0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
      0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
      0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
      -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
      -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,

      -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
      0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
      0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
      0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
      -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
      -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f
  };

  VertexArrayObjectBuilder<float> builder;
  auto vao = builder.stride(6).add(3).add(3).data(vertices, sizeof(vertices)).build();
  if (!vao) {
    std::cerr << "Fail to create vertex array object: " << vao.err_value()
              << std::endl;
    return -1;
  }
  auto &&VAO = vao.take_ok_value();

  auto light_vao = builder.build();
  if (!light_vao) {
    std::cerr << "Fail to create light vao: " << light_vao.err_value()
              << std::endl;
    return -1;
  }

  auto &&light_VAO = light_vao.take_ok_value();

  auto window = ctx->window();
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  glViewport(0, 0, width, height);

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

    shader.use_program();
    shader.set_vec3("object_color", 1.0f, 0.5f, 0.31f);
    shader.set_vec3("light_color", 1.0f, 1.0f, 1.0f);
    shader.set_vec3("light_pos", light_pos);
    shader.set_vec3("view_pos", camera.position());

    glm::mat4 projection =
        glm::perspective(glm::radians(camera.zoom()),
                         (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
    glm::mat4 view = camera.view_matrix();
    glm::mat4 model(1.0f);
    model = glm::rotate(model, 10 * glm::radians(current_time), glm::vec3(0.0f, 1.0f, 0.0f));

    shader.set_mat4("model", model);
    shader.set_mat4("view", view);
    shader.set_mat4("projection", projection);

    VAO.bind();
    glDrawArrays(GL_TRIANGLES, 0, 36);

    light_shader.use_program();

    model = glm::mat4(1.0f);
    light_pos.x = 1.0f + sinf(current_time) * 2.0f;
    light_pos.y = sinf(current_time) / 2.0f;
    model = glm::translate(model, light_pos);
    model = glm::scale(model, glm::vec3(0.2f));
    //model = glm::translate(model, glm::vec3(sinf(current_time), 0.0f, cosf(current_time)));

    light_shader.set_mat4("view", view);
    light_shader.set_mat4("projection", projection);
    light_shader.set_mat4("model", model);

    light_VAO.bind();
    glDrawArrays(GL_TRIANGLES, 0, 36);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

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

  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, true);
  }

  const float camera_speed = 2.5f * delta_time;
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    camera.process_keyboard(CameraMove::Forward, delta_time);
  } else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    camera.process_keyboard(CameraMove::Backward, delta_time);
  } else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
    camera.process_keyboard(CameraMove::Left, delta_time);
  } else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
    camera.process_keyboard(CameraMove::Right, delta_time);
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

  camera.process_mouse_movement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
  camera.process_mouse_scroll(static_cast<float>(yoffset));
}