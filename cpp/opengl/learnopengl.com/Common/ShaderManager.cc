#include <fstream>
#include <iostream>
#include <sstream>
#include "ShaderManager.h"
#include <GL/glew.h>

inline GLenum to_shader_type(shader_type t) {
  switch (t) {
  case shader_type::compute:
    return GL_COMPUTE_SHADER;
  case shader_type::vertex:
    return GL_VERTEX_SHADER;
  case shader_type::tess_control:
    return GL_TESS_CONTROL_SHADER;
  case shader_type::tess_evaluation:
    return GL_TESS_EVALUATION_SHADER;
  case shader_type::geometry:
    return GL_GEOMETRY_SHADER;
  case shader_type::fragment:
    return GL_FRAGMENT_SHADER;
  default:
    return GL_VERTEX_SHADER;
  }
}

Result<GLuint, std::string>
ShaderManager::load(shader_source source, shader_type type, const char *str) {
  const char *data = str;
  std::string file_data;
  if (source == shader_source::from_file) {
    std::ifstream fin;
    fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
      fin.open(str, std::ios::in);
      std::stringstream ss;
      ss << fin.rdbuf();
      file_data = ss.str();
      data = file_data.c_str();
      fin.close();
    } catch (std::system_error &e) {
      return Err(e.code().message());
    }
  }

  GLuint shader = glCreateShader(to_shader_type(type));
  glShaderSource(shader, 1, &data, nullptr);
  glCompileShader(shader);
  int success;
  char info_log[512];
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(shader, sizeof(info_log), nullptr, info_log);
    return Err(std::string(info_log));
  }

  shaders_.push_back(shader);
  return Ok(GLuint(shader));
}
Result<GLuint, std::string> ShaderManager::link() {

  GLuint shader_program = glCreateProgram();
  for (GLuint shader : shaders_) {
    glAttachShader(shader_program, shader);
  }
  glLinkProgram(shader_program);
  int success;
  char info_log[512];
  glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(shader_program, sizeof(info_log), nullptr, info_log);
    return Err(std::string(info_log));
  }

  return Ok(GLuint(shader_program));
}

void ShaderManager::unload_shaders() {
  for (GLuint shader : shaders_) {
    glDeleteShader(shader);
  }
  shaders_.clear();
}

void ShaderManager::unload_program() {
  glDeleteProgram(shader_program_);
  shader_program_ = 0u;
}

ShaderManager::~ShaderManager() {
  unload_shaders();
  unload_program();
}

ShaderManager::ShaderManager(ShaderManager &&o) noexcept {
  if (this != &o) {
    this->shaders_.swap(o.shaders_);
    this->shader_program_ = o.shader_program_;
  }
}

ShaderManager &ShaderManager::operator=(ShaderManager &&o) noexcept {
  if (this != &o) {
    this->shaders_.swap(o.shaders_);
    this->shader_program_ = o.shader_program_;
  }
  return *this;
}
