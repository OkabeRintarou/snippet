#pragma once

#include <string>
#include "Object.h"

class Texture2D : public Object {
public:
  explicit Texture2D(const char * const filename);
  ~Texture2D() noexcept override;
  std::string message() const override;

  void bind();
private:
  std::string error_msg_;
};
