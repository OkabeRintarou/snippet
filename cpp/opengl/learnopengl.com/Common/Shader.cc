#include "Shader.h"
#include <cassert>
#include <fstream>
#include <sstream>

Shader::Shader(shader_source src, const char *vertex_src,
               const char *frag_src) {

  auto create_shader = [&](GLuint &shader, GLenum shader_type, std::string &log,
                           const char *source) -> bool {
    const char *data = source;
    std::string file_data;
    if (src == shader_source::from_file) {
      std::ifstream fin;
      fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);
      try {
        fin.open(source, std::ios::in);
        std::stringstream ss;
        ss << fin.rdbuf();
        file_data = ss.str();
        data = file_data.c_str();
        fin.close();
      } catch (std::system_error &e) {
        log = e.code().message();
        return false;
      }
    }

    shader = glCreateShader(shader_type);
    if (shader == 0 || shader == GL_INVALID_VALUE) {
      log = "Fail to create shader";
      return false;
    }

    glShaderSource(shader, 1, &data, nullptr);
    glCompileShader(shader);
    int success;
    char info_log[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(shader, sizeof(info_log), nullptr, info_log);
      log = std::string(info_log);
      return false;
    }
    return true;
  };

  GLuint vertex_shader, frag_shader;
  if (!create_shader(vertex_shader, GL_VERTEX_SHADER, error_msg_, vertex_src)) {
    return;
  }
  if (!create_shader(frag_shader, GL_FRAGMENT_SHADER, error_msg_, frag_src)) {
    glDeleteShader(vertex_shader);
    return;
  }

  int success;
  char info_log[512];
  GLuint program = glCreateProgram();
  glAttachShader(program, vertex_shader);
  glAttachShader(program, frag_shader);
  glLinkProgram(program);
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(program, sizeof(info_log), nullptr, info_log);
    glDeleteShader(vertex_shader);
    glDeleteShader(frag_shader);
    return;
  }

  vertex_shader_ = vertex_shader;
  frag_shader_ = frag_shader;
  id_ = program;
}

void Shader::use_program() { glUseProgram(id_); }

Shader::~Shader() {
  if (id_ != 0u) {
    glDeleteShader(frag_shader_);
    glDeleteShader(vertex_shader_);
    glDeleteProgram(id_);

    frag_shader_ = vertex_shader_ = id_ = 0u;
  }
}
