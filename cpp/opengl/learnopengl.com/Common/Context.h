#pragma once

#include <optional>
#include <system_error>
#include "Result.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>

class Context {
public:
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&) noexcept;
  Context &operator=(Context &&) noexcept;

  Context() noexcept = default;

  static std::optional<Context> init(int width, int height, const char *title);

  GLFWwindow *window() const { return window_; }
private:
  GLFWwindow *window_ = nullptr;
};
