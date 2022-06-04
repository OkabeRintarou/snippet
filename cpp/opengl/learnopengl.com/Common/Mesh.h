#pragma once

#include "Object.h"
#include "Shader.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <vector>

#define MAX_BONE_INFLUENCE 4

struct Vertex {
  Vertex()
      : position(0.0f), normal(0.0f), tex_coords(0.0f), tangent(0.0f),
        bit_angent(0.0f) {
    for (int i = 0; i < MAX_BONE_INFLUENCE; i++) {
      bone_ids[i] = 0;
      weights[i] = 0.0f;
    }
  }
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec2 tex_coords;
  glm::vec3 tangent;
  glm::vec3 bit_angent;
  int bone_ids[MAX_BONE_INFLUENCE];
  float weights[MAX_BONE_INFLUENCE];
};

struct Texture {
  unsigned id;
  std::string type;
  std::string path;
};

class Mesh : public Object {
public:
  Mesh(std::vector<Vertex> vertices, std::vector<unsigned> indices,
       std::vector<Texture> textures);
  Mesh(Mesh &&o);
  Mesh &operator=(Mesh &&o);

  void draw(Shader &shader);
  bool is_valid() const noexcept override { return error_msg_.empty(); }
  std::string message() const override { return error_msg_; }

private:
  std::vector<Vertex> vertices_;
  std::vector<unsigned> indices_;
  std::vector<Texture> textures_;
  unsigned int VAO, VBO, EBO;
  std::string error_msg_;

private:
  void setup();
};