#include "util.h"
/* clang-format off */
#include <GL/glew.h>
#include <GL/freeglut.h>
/* clang-format on */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

GLuint VBO;
GLuint TBO;

static void CreateVertexBuffer() {
    float vertices[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glGenBuffers(1, &TBO);
    glBindBuffer(GL_ARRAY_BUFFER, TBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), nullptr, GL_STATIC_READ);
}

const char *vs_filename = "shaders/transform-feedback/example1.vert";

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

    std::string vs;

    if (!ReadFile(vs_filename, vs)) {
        fprintf(stderr, "Error read file %s\n", vs_filename);
        exit(-1);
    }
    AddShader(shader_program, vs.c_str(), GL_VERTEX_SHADER);

    // specify feedback variables
    const GLchar *feedback_varyings[] = {"outValue"};
    glTransformFeedbackVaryings(shader_program, 1, feedback_varyings, GL_INTERLEAVED_ATTRIBS);

    glLinkProgram(shader_program);

    GLint success = 0;
    GLchar info_log[1024] = {0};
    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shader_program, sizeof(info_log), nullptr, info_log);
        fprintf(stderr, "Error linking shader program: '%s'", info_log);
        exit(-1);
    }

    glValidateProgram(shader_program);
    glGetProgramiv(shader_program, GL_VALIDATE_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shader_program, sizeof(info_log), nullptr, info_log);
        fprintf(stderr, "Invalid shader: '%s'", info_log);
        exit(-1);
    }
    glUseProgram(shader_program);
}

int main(int argc, char *argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(1024, 768);
    glutCreateWindow("Transform Feedback Example1");

    auto res = glewInit();
    if (res != GLEW_OK) {
        fprintf(stderr, "Error: '%s'\n", glewGetErrorString(res));
        return -1;
    }

    CreateVertexBuffer();
    CompileShader();

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, TBO);
    // Perform feedback transform
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 5);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);

    glFlush();

    // Fetch and print results
    float feedback[5] = {0.0f};
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(feedback), feedback);

    printf("%f %f %f %f %f\n", feedback[0], feedback[1], feedback[2], feedback[3], feedback[4]);
    return 0;
}
