#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include "Texture2D.h"

Texture2D::Texture2D(const char *const filename) {
  GLuint id;
  glGenTextures(1, &id);
  glBindTexture(GL_TEXTURE_2D, id);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  int width, height, channels;
  unsigned char *img_data = stbi_load(filename, &width, &height, &channels, 0);
  if (!img_data) {
    error_msg_ = std::string("Fail to load image ") + filename;
    glDeleteTextures(1, &id);
    return;
  }
  GLenum format = GL_RGB;
  if (channels == 1) {
    format = GL_RED;
  } else if (channels == 4) {
    format = GL_RGBA;
  }

  glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format,
               GL_UNSIGNED_BYTE, img_data);
  glGenerateMipmap(GL_TEXTURE_2D);
  stbi_image_free(img_data);

  id_ = id;
}

Texture2D::~Texture2D() noexcept {
  glDeleteTextures(1, &id_);
  id_ = 0;
}
std::string Texture2D::message() const {
  if (is_valid()) {
    return Object::message();
  }
  return error_msg_;
}

void Texture2D::bind(int tex_index) {
  glActiveTexture(GL_TEXTURE0 + tex_index);
  glBindTexture(GL_TEXTURE_2D, id_);
}