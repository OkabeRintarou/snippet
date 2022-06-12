
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

int main() {
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
                "../../Resources/Shader/StencilTest_Vertex.glsl",
                "../../Resources/Shader/StencilTest_Frag.glsl");
  if (!shader.is_valid()) {
    std::cerr << "Fail to create shader: " << shader.message() << std::endl;
    return -1;
  }

  Shader single_color_shader(shader_source::from_file,
                      "../../Resources/Shader/StencilTest_Vertex.glsl",
                      "../../Resources/Shader/Stencil_SingleColor_Frag.glsl");
  if (!single_color_shader.is_valid()) {
    std::cerr << "Fail to create light shader: " << single_color_shader.message()
              << std::endl;
    return -1;
  }

  Texture2D cube_texture("../../Resources/Textures/marble.jpg");
  if (!cube_texture.is_valid()) {
    std::cerr << "Fail to load cube texture: " << cube_texture.message() << std::endl;
    return -1;
  }

  Texture2D floor_texture("../../Resources/Textures/metal.png");
  if (!floor_texture.is_valid()) {
    std::cerr << "Fail to load floor texture: " << floor_texture.message() << std::endl;
    return -1;
  }

  // set up vertex data (and buffer(s)) and configure vertex attributes
  // ------------------------------------------------------------------
  float cube_vertices[] = {
      // positions          // texture Coords
      -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
      0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
      0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
      0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
      -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
      -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

      -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
      0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
      0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
      0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
      -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
      -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

      -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
      -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
      -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
      -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
      -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
      -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

      0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
      0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
      0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
      0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
      0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
      0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

      -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
      0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
      0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
      0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
      -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
      -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

      -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
      0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
      0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
      0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
      -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
      -0.5f,  0.5f, -0.5f,  0.0f, 1.0f
  };
  float plane_vertices[] = {
      // positions          // texture Coords (note we set these higher than 1 (together with GL_REPEAT as texture wrapping mode). this will cause the floor texture to repeat)
      5.0f, -0.5f,  5.0f,  2.0f, 0.0f,
      -5.0f, -0.5f,  5.0f,  0.0f, 0.0f,
      -5.0f, -0.5f, -5.0f,  0.0f, 2.0f,

      5.0f, -0.5f,  5.0f,  2.0f, 0.0f,
      -5.0f, -0.5f, -5.0f,  0.0f, 2.0f,
      5.0f, -0.5f, -5.0f,  2.0f, 2.0f
  };

  VertexArrayObjectBuilder<float> builder;
  auto vao = builder.stride(5).add(3).add(2).data(cube_vertices, sizeof(cube_vertices)).build();
  if (!vao) {
    std::cerr << "Fail to create vertex array object: " << vao.err_value()
              << std::endl;
    return -1;
  }
  auto &&VAO = vao.take_ok_value();

  VertexArrayObjectBuilder<float> plane_builder;
  auto plane_vao = plane_builder.stride(5).add(3).add(2).data(plane_vertices, sizeof(plane_vertices)).build();
  if (!plane_vao) {
    std::cerr << "Fail to create plane vao: " << plane_vao.err_value()
              << std::endl;
    return -1;
  }

  auto &&plane_VAO = plane_vao.take_ok_value();

  auto window = ctx->window();
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  glViewport(0, 0, width, height);

  // configure global opengl state
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);

  glEnable(GL_STENCIL_TEST);
  glStencilFunc(GL_NOTEQUAL, 1, 0xff);
  glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

  shader.set_int("texture1", 0);
  single_color_shader.set_int("texture1", 0);

  // render loop
  // -----------
  while (!glfwWindowShouldClose(window)) {

    auto current_time = static_cast<float>(glfwGetTime());
    delta_time = current_time - last_frame;
    last_frame = current_time;

    processInput(window);

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    glm::mat4  projection = glm::perspective(glm::radians(camera.zoom()),
                     (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
    glm::mat4 view = camera.view_matrix();
    glm::mat4 model(1.0f);


    shader.use_program();

    // draw floor
    glStencilMask(0x00);
    shader.set_mat4("model", model);
    plane_VAO.bind();
    floor_texture.bind(0);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // 1st render pass
    glStencilFunc(GL_ALWAYS, 1, 0xff);
    glStencilMask(0xff);

    shader.set_mat4("view", view);
    shader.set_mat4("projection", projection);

    VAO.bind();
    cube_texture.bind(0);

    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(-1.0f, 0.0f, -1.0f));
    shader.set_mat4("model", model);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(2.0f, 0.0f, 0.0f));
    shader.set_mat4("model", model);
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 2nd render pass
    single_color_shader.use_program();
    single_color_shader.set_mat4("view", view);
    single_color_shader.set_mat4("projection", projection);
    single_color_shader.set_mat4("model", glm::mat4(1.0f));

    glDisable(GL_DEPTH_TEST);
    glStencilFunc(GL_NOTEQUAL, 1, 0xff);
    glStencilMask(0x00);

    const float scale = 1.1f;
    VAO.bind();
    cube_texture.bind(0);

    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(-1.0f, 0.0f, -1.0f));
    model = glm::scale(model, glm::vec3(scale, scale, scale));
    single_color_shader.set_mat4("model", model);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(2.0f, 0.0f, 0.0f));
    model = glm::scale(model, glm::vec3(scale, scale, scale));
    single_color_shader.set_mat4("model", model);
    glDrawArrays(GL_TRIANGLES, 0, 36);

    glEnable(GL_DEPTH_TEST);
    glStencilFunc(GL_ALWAYS, 0, 0xff);
    glStencilMask(0xff);

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
