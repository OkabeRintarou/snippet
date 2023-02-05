#version 330 core

layout (location = 0) in vec3 Position;

uniform mat4 gTranslation;

out vec4 Color;

void main()
{
    gl_Position = gTranslation * vec4(Position, 1.0f);
    Color = vec4(clamp(Position, 0.0f, 1.0f), 1.0f);
}