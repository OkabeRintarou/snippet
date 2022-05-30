#pragma once

#include "Result.h"
#include "gl_header.h"
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

class Object {
public:
  Object() = default;
  virtual ~Object() { id_ = 0u; };

  Object(const Object &) = delete;
  Object &operator=(const Object &) = delete;

  Object(Object &&o) noexcept {
    this->id_ = o.id_;
    o.id_ = 0u;
  }
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

class VertexArrayObject : public Object {
public:
  template <typename T> friend class VertexArrayObjectBuilder;
  void bind();

  VertexArrayObject() : vbo_(0u) {}
  ~VertexArrayObject() override {
    if (id_ != 0u) {
      glDeleteVertexArrays(1, &id_);
      glDeleteBuffers(1, &vbo_);
    }

    id_ = vbo_ = 0u;
  }
  VertexArrayObject(VertexArrayObject &&o) noexcept {
    id_ = o.id_;
    vbo_ = o.vbo_;
    o.id_ = o.vbo_ = 0u;
  }

  VertexArrayObject &operator=(VertexArrayObject &&o) {
    if (this != &o) {
      vbo_ = o.vbo_;
      Object::operator=(std::move(o));
    }
    return *this;
  }

private:
  GLuint vbo_;
};

namespace details {
template <typename U> struct enum_type;

template <> struct enum_type<float> {
  static constexpr GLenum value = GL_FLOAT;
};

template <> struct enum_type<char> { static constexpr GLenum value = GL_BYTE; };

template <> struct enum_type<unsigned char> {
  static constexpr GLenum value = GL_UNSIGNED_BYTE;
};

template <> struct enum_type<short> {
  static constexpr GLenum value = GL_SHORT;
};

template <> struct enum_type<unsigned short> {
  static constexpr GLenum value = GL_UNSIGNED_SHORT;
};

template <> struct enum_type<int> { static constexpr GLenum value = GL_INT; };

template <> struct enum_type<unsigned int> {
  static constexpr GLenum value = GL_UNSIGNED_INT;
};
} // namespace details

template <typename T> class VertexArrayObjectBuilder {
public:
  VertexArrayObjectBuilder() = default;

  VertexArrayObjectBuilder(const VertexArrayObjectBuilder &) = delete;
  void operator=(const VertexArrayObjectBuilder &) = delete;
  VertexArrayObjectBuilder(VertexArrayObjectBuilder &&) = delete;
  void operator=(VertexArrayObjectBuilder &&) = delete;

  VertexArrayObjectBuilder &stride(GLsizei s) {
    stride_.emplace(s * sizeof(T));
    return *this;
  }

  VertexArrayObjectBuilder &add(GLint size, bool normalized = false) {
    GLenum type = details::enum_type<T>::value;
    configs_.emplace_back(type, size, (const void *)(offset_ * sizeof(T)),
                          normalized);
    offset_ += size;
    return *this;
  }

  VertexArrayObjectBuilder &data(const T *value, std::size_t bytes) {
    data_ = value;
    // data_.emplace(value);
    data_size_ = bytes;
    return *this;
  }

  Result<VertexArrayObject, std::string> build(GLenum usage = GL_STATIC_DRAW) {
    VertexArrayObject empty;
    GLuint vao = 0u, vbo = 0u;

    if (!stride_) {
      return make_err<VertexArrayObject, std::string>(
          std::string("Not set stride"));
    }
    if (!data_) {
      return make_err<VertexArrayObject, std::string>(
          std::string("Not set buffer data"));
    }
    glGenVertexArrays(1, &vao);
    if (vao == 0 || vao == GL_INVALID_VALUE) {
      return make_err<VertexArrayObject, std::string>(
          std::string("Fail to gen vertex arrays"));
    }

    glGenBuffers(1, &vbo);
    if (vbo == 0 || vbo == GL_INVALID_VALUE) {
      return make_err<VertexArrayObject, std::string>(
          std::string("Fail to gen buffers"));
    }

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, data_size_, *data_, usage);

    for (auto i = 0, e = configs_.size(); i < e; i++) {
      const auto &conf = configs_[i];
      printf("glVertexAttribPointer(%d, %d, %d, %d, %d, %p), GL_FLOAT = %d\n",
             i, conf.size, conf.type, conf.normalized, *stride_, conf.offset,
             GL_FLOAT);
      glVertexAttribPointer((GLuint)i, conf.size, conf.type, conf.normalized,
                            *stride_, conf.offset);
      glEnableVertexAttribArray(i);
    }
    VertexArrayObject obj;
    obj.id_ = vao;
    obj.vbo_ = vbo;
    return make_ok<VertexArrayObject, std::string>(std::move(obj));
  }

private:
  GLuint offset_ = 0u;
  std::optional<GLsizei> stride_;
  std::optional<const T *> data_;
  std::size_t data_size_ = 0;

  struct Config {
    Config(GLenum t, GLint s, const void *o, bool n = false)
        : type(t), size(s), offset(o), normalized(n) {}
    GLenum type;
    GLint size;
    const void *offset;
    bool normalized;
  };
  std::vector<Config> configs_;
};