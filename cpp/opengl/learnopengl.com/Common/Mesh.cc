#include <cassert>
#include <cstddef>
#include "Mesh.h"
#include "Object.h"

Mesh::Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices,
           std::vector<Texture> textures)
    : vertices_(std::move(vertices)), indices_(std::move(indices)),
      textures_(std::move(textures)) {
  setup();
}

void Mesh::setup() {
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);

  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);

  glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(Vertex), &vertices_[0], GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof(unsigned), &indices_[0], GL_STATIC_DRAW);

  // vertex positions
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));
  // vertex normals
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));

  // vertex texture coords
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tex_coords));

  glBindVertexArray(0u);
}

void Mesh::draw(Shader &shader) {
  unsigned diff_nr = 1;
  unsigned specular_nr = 1;

  for (unsigned i = 0; i < textures_.size(); i++) {
    glActiveTexture(GL_TEXTURE0 + i);
    std::string number;
    const std::string &name = textures_[i].type;
    if (name == "texture_diffuse") {
      number = std::to_string(diff_nr++);
    } else if (name == "texture_specular") {
      number = std::to_string(specular_nr++);
    } else {
      assert(false && "should never reach here");
    }

    std::string param = std::string("material.");
    param += name + number;
    shader.set_int(param.c_str(), (int)i);
    glBindTexture(GL_TEXTURE_2D, textures_[i].id);
  }

  glActiveTexture(GL_TEXTURE0);

  // draw mesh
  glBindVertexArray(VAO);
  glDrawElements(GL_TRIANGLES, indices_.size(), GL_UNSIGNED_INT, 0);
  glBindVertexArray(0u);
}

Mesh::Mesh(Mesh &&o) {
  this->vertices_.swap(o.vertices_);
  this->indices_.swap(o.indices_);
  this->textures_.swap(o.textures_);
  this->VAO = o.VAO;
  this->VBO = o.VBO;
  this->EBO = o.EBO;
  this->error_msg_.swap(o.error_msg_);
}

Mesh& Mesh::operator=(Mesh &&o) {
  if (this != &o) {
    this->vertices_.swap(o.vertices_);
    this->indices_.swap(o.indices_);
    this->textures_.swap(o.textures_);
    this->VAO = o.VAO;
    this->VBO = o.VBO;
    this->EBO = o.EBO;
    this->error_msg_.swap(o.error_msg_);
  }
  return *this;
}
