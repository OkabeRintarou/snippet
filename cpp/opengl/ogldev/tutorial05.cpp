#include "util.h"
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

GLuint VBO;
GLint gScaleLocation;

static void RenderScreenCB() {
    static float scale = 0.0f;
    static float delta = 0.001f;

    scale += delta;
    if ((scale >= 1.0f) || (scale <= -1.0f)) {
        delta *= -1.0f;
    }
    glUniform1f(gScaleLocation, scale);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *) 0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisableVertexAttribArray(0);

    glutPostRedisplay();
    glutSwapBuffers();
}

static void CreateVertexBuffer() {
    float vertices[] = {-1.0f, -1.0f, 0.0f,// bottom left
                        1.0f, -1.0f, 0.0f, // bottom right
                        1.0f, 1.0f, 0.0f}; // top right

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
}

const char *vs_filename = "shaders/tutorial05/shader.vert";
const char *ps_filename = "shaders/tutorial05/shader.frag";

static void AddShader(GLuint program, const char *text, GLenum type) {
    GLuint shader = glCreateShader(type);
    if (shader == 0) {
        fprintf(stderr, "Error creating shader type %d\n", type);
        exit(-1);
    }
    const char *p[1];
    p[0] = text;
    GLint lengths[1];
    lengths[0] = static_cast<GLint>(strlen(text));
    glShaderSource(shader, 1, p, lengths);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar info_log[1024];
        glGetShaderInfoLog(shader, 1024, nullptr, info_log);
        fprintf(stderr, "Error compiling shader type %d: %s\n", type, info_log);
        exit(-1);
    }
    glAttachShader(program, shader);
}

static void CompileShader() {
    GLuint shader_program = glCreateProgram();
    if (shader_program == 0) {
        fprintf(stderr, "Error create shader program\n");
        exit(-1);
    }

    std::string vs, ps;

    if (!ReadFile(vs_filename, vs)) {
        fprintf(stderr, "Error read file %s\n", vs_filename);
        exit(-1);
    }
    AddShader(shader_program, vs.c_str(), GL_VERTEX_SHADER);

    if (!ReadFile(ps_filename, ps)) {
        fprintf(stderr, "Error read file %s\n", ps_filename);
        exit(-1);
    }
    AddShader(shader_program, ps.c_str(), GL_FRAGMENT_SHADER);

    glLinkProgram(shader_program);

    GLint success = 0;
    GLchar info_log[1024] = {0};
    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shader_program, sizeof(info_log), nullptr, info_log);
        fprintf(stderr, "Error linking shader program: '%s'", info_log);
        exit(-1);
    }

    gScaleLocation = glGetUniformLocation(shader_program, "gScale");
    if (gScaleLocation == -1) {
        fprintf(stderr, "Error getting uniform location of 'gScale'\n");
        exit(-1);
    }

    glValidateProgram(shader_program);
    glGetProgramiv(shader_program, GL_VALIDATE_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shader_program, sizeof(info_log), nullptr, info_log);
        fprintf(stderr, "Invalid sahder: '%s'", info_log);
        exit(-1);
    }
    glUseProgram(shader_program);
}

int main(int argc, char *argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(1024, 768);
    glutCreateWindow("Tutorial 05");
    glutDisplayFunc(RenderScreenCB);

    auto res = glewInit();
    if (res != GLEW_OK) {
        fprintf(stderr, "Error: '%s'\n", glewGetErrorString(res));
        return -1;
    }

    CreateVertexBuffer();
    CompileShader();

    glClearColor(.0f, .0f, .0f, .0f);
    glutMainLoop();
    return 0;
}
