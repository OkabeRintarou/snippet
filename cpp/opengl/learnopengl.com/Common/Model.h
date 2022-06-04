#pragma once

#include "Mesh.h"
#include "Object.h"
#include "Shader.h"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <string>
#include <vector>
#include <unordered_set>

class Model : public Object {
public:
  Model(const char *path);
  void draw(Shader &shader);

  bool is_valid() const noexcept override { return error_msg_.empty(); }
  std::string message() const override { return error_msg_; }

private:
  void load_model(const std::string &path);
  void process_node(aiNode *node, const aiScene *scene);
  Mesh process_mesh(aiMesh *mesh, const aiScene *scene);
  std::vector<Texture> load_material_textures(aiMaterial *mat,
                                              aiTextureType type,
                                              const std::string &type_name);
  bool is_texture_loaded(const char *name) const;
private:
  std::unordered_set<std::string> texture_loaded_;
  std::vector<Mesh> meshes_;
  std::string directory_;
  std::string error_msg_;
};
