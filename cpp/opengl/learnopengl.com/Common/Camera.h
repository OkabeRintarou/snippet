#pragma once

#include <cstdint>
#include <cassert>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

enum class CameraMove : uint8_t {
  Forward,
  Backward,
  Left,
  Right,
};

static const float YAW = -90.0f;
static const float PITCH = 0.0f;
static const float SPEED = 2.5f;
static const float SENSITIVITY = 0.1f;
static const float ZOOM = 45.0f;

class Camera {
public:
  Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f),
         glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW,
         float pitch = PITCH) {
    position_ = position;
    up_ = up;
    yaw_ = yaw;
    pitch_ = pitch;
    init();
    update_camera_vectors();
  }

  Camera(float pos_x, float pos_y, float pos_z, float up_x, float up_y, float up_z, float yaw = YAW, float pitch = PITCH) {
    position_ = glm::vec3(pos_x, pos_y, pos_z);
    up_ = glm::vec3(up_x, up_y, up_z);
    yaw_ = yaw;
    pitch_ = pitch;
    init();
    update_camera_vectors();
  }

  glm::mat4 view_matrix() const {
    return glm::lookAt(position_, position_ + front_, up_);
  }

  void process_keyboard(CameraMove direction, float delta_time) {
    float velocity = movement_speed_ * delta_time;
    switch (direction) {
    case CameraMove::Forward:
      position_ += front_ * velocity;
    case CameraMove::Backward:
      position_ -= front_ * velocity;
    case CameraMove::Left:
      position_ -= right_ * velocity;
    case CameraMove::Right:
      position_ += right_ * velocity;
    default:
      assert(false && "never reach here");
    }
  }
private:
  void init() {
    front_ = glm::vec3(0.0f, 0.0f, -1.0f);
    movement_speed_ = SPEED;
    mouse_sensitivity_ = SENSITIVITY;
    zoom_ = ZOOM;
  }
  void update_camera_vectors() {
    glm::vec3 front;
    front.x = cosf(glm::radians(yaw_)) * cosf(glm::radians(pitch_));
    front.y = sinf(glm::radians(pitch_));
    front.z = sinf(glm::radians(yaw_)) * cosf(glm::radians(pitch_));
    front_ = glm::normalize(front);
    right_ = glm::normalize(glm::cross(front_, up_));
    up_ = glm::normalize(glm::cross(right_, front_));
  }

private:
  glm::vec3 position_;
  glm::vec3 front_;
  glm::vec3 up_;
  glm::vec3 right_;
  glm::vec3 world_up_;
  float yaw_;
  float pitch_;
  float movement_speed_;
  float mouse_sensitivity_;
  float zoom_;
};
