#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Model.h"

static unsigned TextureFromFile(const char *path, const std::string &directory, bool gamma, std::string &err) {
  std::string filename = std::string(path);
  filename = directory + "/" + filename;

  unsigned texture_id;
  glGenTextures(1, &texture_id);

  int width, height, channels;
  unsigned char *data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
  if (!data) {
    err = "Fail to load texture: " + filename;
    return 0u;
  }

  GLenum format;
  switch (channels) {
  case 1:
    format = GL_RED;
    break;
  case 3:
    format = GL_RGB;
    break;
  case 4:
    format = GL_RGBA;
    break;
  default:
    format = 0;
    break;
  }
  glBindTexture(GL_TEXTURE_2D, texture_id);
  glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
  glGenerateMipmap(GL_TEXTURE_2D);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  stbi_image_free(data);
  return texture_id;
}

Model::Model(const char *path) {
  stbi_set_flip_vertically_on_load(true);
  load_model(path);
}

void Model::draw(Shader &shader) {
  for (auto &&mesh : meshes_) {
    mesh.draw(shader);
  }
}

void Model::load_model(const std::string &path) {
  Assimp::Importer importer;
  const aiScene *scene =
      importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);
  if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE ||
      !scene->mRootNode) {
    error_msg_ = importer.GetErrorString();
    return;
  }
  directory_ = path.substr(0, path.find_last_of('/'));
  process_node(scene->mRootNode, scene);
}

void Model::process_node(aiNode *node, const aiScene *scene) {
  for (unsigned i = 0; i < node->mNumMeshes; i++) {
    aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
    meshes_.push_back(process_mesh(mesh, scene));
  }

  for (unsigned i = 0; i < node->mNumChildren; i++) {
    process_node(node->mChildren[i], scene);
  }
}

Mesh Model::process_mesh(aiMesh *mesh, const aiScene *scene) {
  std::vector<Vertex> vertices;
  std::vector<unsigned> indices;
  std::vector<Texture> textures;

  for (unsigned i = 0; i < mesh->mNumVertices; i++) {
    Vertex vertex;
    glm::vec3 v;
    v.x = mesh->mVertices[i].x;
    v.y = mesh->mVertices[i].y;
    v.z = mesh->mVertices[i].z;
    vertex.position = v;

    if (mesh->HasNormals()) {
      glm::vec3 normal;
      normal.x = mesh->mNormals[i].x;
      normal.y = mesh->mNormals[i].y;
      normal.z = mesh->mNormals[i].z;
      vertex.normal = normal;
    }

    glm::vec2 tex_coords{0.0f, 0.0f};
    if (mesh->mTextureCoords[0]) {
      tex_coords.x = mesh->mTextureCoords[0][i].x;
      tex_coords.y = mesh->mTextureCoords[0][i].y;

//      v.x = mesh->mTangents[i].x;
//      v.y = mesh->mTangents[i].y;
//      v.z = mesh->mTangents[i].z;
//      vertex.tangent = v;
//
//      v.x = mesh->mBitangents[i].x;
//      v.y = mesh->mBitangents[i].y;
//      v.z = mesh->mBitangents[i].z;
//      vertex.bit_angent = v;
    }
    vertex.tex_coords = tex_coords;

    vertices.emplace_back(vertex);
  }

  for (unsigned i = 0; i < mesh->mNumFaces; i++) {
    const aiFace &face = mesh->mFaces[i];
    for (unsigned j = 0; j < face.mNumIndices; j++) {
      indices.push_back(face.mIndices[j]);
    }
  }

  if (mesh->mMaterialIndex >= 0) {
    aiMaterial *material = scene->mMaterials[mesh->mMaterialIndex];
    std::vector<Texture> diffuse_maps = load_material_textures(material, aiTextureType_DIFFUSE, "texture_diffuse");
    textures.insert(textures.end(), diffuse_maps.begin(), diffuse_maps.end());
    std::vector<Texture> specular_maps = load_material_textures(material, aiTextureType_SPECULAR, "texture_specular");
    textures.insert(textures.end(), specular_maps.begin(), specular_maps.end());
    std::vector<Texture> normal_maps = load_material_textures(material, aiTextureType_HEIGHT, "texture_normal");
    textures.insert(textures.end(), normal_maps.begin(), normal_maps.end());
    std::vector<Texture> height_maps = load_material_textures(material, aiTextureType_AMBIENT, "texture_height");
    textures.insert(textures.end(), height_maps.begin(), height_maps.end());
  }
  return Mesh{std::move(vertices), std::move(indices), std::move(textures)};
}

std::vector<Texture>
Model::load_material_textures(aiMaterial *mat, aiTextureType type,
                              const std::string &type_name) {
  std::vector<Texture> ret;
  for (unsigned i = 0; i < mat->GetTextureCount(type); i++) {
    aiString str;
    mat->GetTexture(type, i, &str);
    // check if texture was loaded before and if so,
    // continue to next iteration: skip loading a new texture
    if (!is_texture_loaded(str.C_Str())) {
      Texture texture;
      texture.id = TextureFromFile(str.C_Str(), directory_, false, error_msg_);
      if (texture.id == 0u || !error_msg_.empty()) {
        break;
      }
      texture.type = type_name;
      texture.path = str.C_Str();
      ret.push_back(texture);

      texture_loaded_.emplace(str.C_Str());
    }
  }
  return ret;
}

bool Model::is_texture_loaded(const char *name) const {
  return texture_loaded_.find(name) != texture_loaded_.end();
}
