#pragma once

#include <cassert>
#include "Object.h"
#include "Result.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

enum class shader_source {
  from_string,
  from_file,
};

class Shader : public Object {
public:
  Shader(shader_source src, const char *vertex_src, const char *frag_src);
  ~Shader();
  void use_program();

  std::string message() const override {
    if (!is_valid()) {
         return error_msg_;
    }
    return Object::message();
  }

  void set_bool(const std::string &name, bool value) const {
    set_bool(name.c_str(), value);
  }

  void set_bool(const char *name, bool value) const {
    GLint loc = glGetUniformLocation(id_, name);
    assert(loc != GL_INVALID_VALUE && loc != GL_INVALID_OPERATION);
    glUniform1i(loc, value ? 1 : 0);
  }

  void set_int(const std::string &name, int value) const {
    set_int(name.c_str(), value);
  }

  void set_int(const char *name, int value) const {
    GLint loc = glGetUniformLocation(id_, name);
    assert(loc != GL_INVALID_VALUE && loc != GL_INVALID_OPERATION);
    glUniform1i(loc, value);
  }

  void set_float(const std::string &name, float value) const {
    set_int(name.c_str(), value);
  }

  void set_float(const char *name, float value) const {
    GLint loc = glGetUniformLocation(id_, name);
    assert(loc != GL_INVALID_VALUE && loc != GL_INVALID_OPERATION);
    glUniform1f(loc, value);
  }

  void set_vec2(const char *name, const glm::vec2 &vec2) {
    GLint loc = glGetUniformLocation(id_, name);
    assert(loc != GL_INVALID_VALUE && loc != GL_INVALID_OPERATION);
    glUniform2fv(loc, 1, glm::value_ptr(vec2));
  }

  void set_vec3(const char *name, float v1, float v2, float v3) {
    GLint loc = glGetUniformLocation(id_, name);
    assert(loc != GL_INVALID_VALUE && loc != GL_INVALID_OPERATION);
    glUniform3f(loc, v1, v2, v3);
  }

  void set_vec3(const char *name, const glm::vec3 &vec) {
    GLint loc = glGetUniformLocation(id_, name);
    assert(loc != GL_INVALID_VALUE && loc != GL_INVALID_OPERATION);
    glUniform3f(loc, vec[0], vec[1], vec[2]);
  }

  void set_mat4(const char *name, const glm::mat4 &mat4) {
    GLint loc = glGetUniformLocation(id_, name);
    assert(loc != GL_INVALID_VALUE && loc != GL_INVALID_OPERATION);
    glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(mat4));
  }
private:
  std::string error_msg_;
  GLuint vertex_shader_ = 0u;
  GLuint frag_shader_ = 0u;
};
