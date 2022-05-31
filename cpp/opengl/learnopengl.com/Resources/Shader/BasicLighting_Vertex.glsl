#version 330 core

out vec3 frag_pos;
out vec3 normal;

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;


void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0f);
    frag_pos = vec3(model * vec4(aPos, 1.0f));
    normal = vec3(model * vec4(aNormal, 0.0f));
}