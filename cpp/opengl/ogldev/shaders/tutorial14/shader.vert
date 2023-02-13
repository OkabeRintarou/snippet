#version 330 core

layout (location = 0) in vec3 Position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec4 Color;

void main()
{
    gl_Position =  projection * view * model * vec4(Position, 1.0f);
    Color = vec4(clamp(Position, 0.0f, 1.0f), 1.0f);
}