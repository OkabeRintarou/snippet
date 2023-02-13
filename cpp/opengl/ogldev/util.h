#pragma once

#include <string>
#include <glm/glm.hpp>

bool ReadFile(const char *filename, std::string &content);

class Pipeline {
public:
    Pipeline() {
        scale_ = glm::vec3(1.0f, 1.0f, 1.0f);
        world_pos_ = glm::vec3(0.0f, 0.0f, 0.0f);
        rotate_info_ = glm::vec3(0.0f, 0.0f, 0.0f);
    }

    void Scale(float s) {
        Scale(s, s, s);
    }

    void Scale(const glm::vec3 &v) {
        Scale(v.x, v.y, v.z);
    }

    void Scale(float x, float y, float z) {
        scale_.x = x;
        scale_.y = y;
        scale_.z = z;
    }

    void WorldPos(float x, float y, float z) {
        world_pos_.x = x;
        world_pos_.y = y;
        world_pos_.z = z;
    }

    void WorldPos(const glm::vec3 &v) {
        world_pos_ = v;
    }

    void Rotate(float x, float y, float z) {
        rotate_info_.x = x;
        rotate_info_.y = y;
        rotate_info_.z = z;
    }

    void Rotate(const glm::vec3 &v) {
        rotate_info_ = v;
    }
private:
    glm::vec3 scale_;
    glm::vec3 world_pos_;
    glm::vec3 rotate_info_;
};
