#version 330 core

uniform float gScale;
layout (location = 0) in vec3 Position;

void main()
{
    gl_Position = vec4(0.5 * gScale * Position.x, 0.5 * gScale * Position.y, Position.z, 1.0f);
}