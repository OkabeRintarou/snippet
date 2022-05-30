#include "Context.h"
#include <iostream>

std::optional<Context> Context::init(int width, int height, const char *title) {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

  GLFWwindow *window =
      glfwCreateWindow(width, height, "LearnOpenGL", nullptr, nullptr);
  if (window == nullptr) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return std::nullopt;
  }

  glfwMakeContextCurrent(window);

  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    std::cerr << "Failed to initialize GLEW" << std::endl;
    return std::nullopt;
  }
  std::optional<Context> ctx{std::in_place};
  ctx->window_ = window;

  return std::optional<Context>{std::move(ctx)};
}

Context::Context(Context &&o) noexcept {
  if (this != &o) {
    this->window_ = o.window_;
  }
}

Context &Context::operator=(Context &&o) noexcept {
  if (this != &o) {
    this->window_ = o.window_;
  }
  return *this;
}
