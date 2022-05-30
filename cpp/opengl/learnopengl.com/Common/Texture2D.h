#pragma once

#include <string>
#include "Object.h"

class Texture2D : public Object {
public:
  explicit Texture2D(const char * const filename);
  ~Texture2D() noexcept override;
  std::string message() const override;

  Texture2D(const Texture2D&) = delete;
  void operator=(const Texture2D &) = delete;
  Texture2D(Texture2D &&) = delete;
  void operator=(Texture2D &&) = delete;

  void bind(int tex_index);
private:
  std::string error_msg_;
};
