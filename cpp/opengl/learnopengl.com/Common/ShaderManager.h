#pragma once

#include "Result.h"
#include <string>
#include <vector>
#include <GL/glew.h>

enum class shader_source {
  from_string,
  from_file,
};

enum class shader_type {
  compute,
  vertex,
  tess_control,
  tess_evaluation,
  geometry,
  fragment,
};

class ShaderManager {
public:
  ShaderManager() noexcept = default;
  ~ShaderManager();
  ShaderManager(const ShaderManager&) = delete;
  ShaderManager& operator=(const ShaderManager&) = delete;

  ShaderManager(ShaderManager &&) noexcept;
  ShaderManager& operator=(ShaderManager &&) noexcept;

  Result<GLuint, std::string> load(shader_source, shader_type,
                                       const char *str);
  void unload_shaders();
  Result<GLuint, std::string> link();
  void unload_program();
  const GLuint shader_program() const { return shader_program_; }
private:
  std::vector<GLuint> shaders_;
  GLuint shader_program_ = 0u;
};
