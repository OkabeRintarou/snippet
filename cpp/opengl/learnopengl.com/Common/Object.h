#pragma once

#include <string>
#include <vector>
#include "gl_header.h"

class Object {
public:
  Object() = default;
  virtual ~Object() = default;

  Object(const Object&) = delete;
  Object &operator=(const Object &) = delete;

  Object(Object && o) noexcept { this->id_ = o.id_; o.id_ = 0u; }
  Object &operator=(Object &&o) noexcept {
    GLuint id = o.id_;
    o.id_ = 0u;
    this->id_ = id;
    return *this;
  }

  GLuint id() const noexcept { return id_; }
  virtual bool is_valid() const noexcept { return id_ != 0; };
  virtual std::string message() const { return "SUCC"; }
protected:
  GLuint id_ = 0;
};
