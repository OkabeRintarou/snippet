#include "util.h"
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

GLuint VBO;
GLuint IBO;
GLint gModelLocation;
GLint gViewLocation;
GLint gProjectionLocation;

static void RenderScreenCB() {
    const float radius = 10.0f;
    auto elapsed = static_cast<float>(glfwGetTime());
    float cam_x = sinf(elapsed) * radius;
    float cam_z = cosf(elapsed) * radius;

    // camera rotate around y-axis
    glm::mat4 view = glm::lookAt(glm::vec3(cam_x, 0.0f, cam_z), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glUniformMatrix4fv(gViewLocation, 1, GL_FALSE, glm::value_ptr(view));

    glm::mat4 model(1.0f);
    glUniformMatrix4fv(gModelLocation, 1, GL_FALSE, glm::value_ptr(model));

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *) 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);

    glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, static_cast<void*>(0));

    glDisableVertexAttribArray(0);

    glutPostRedisplay();
    glutSwapBuffers();
}

static void InitializeGlutCallbacks() {
    glutDisplayFunc(RenderScreenCB);
    glutIdleFunc(RenderScreenCB);
}

static void CreateVertexBuffer() {
    float vertices[] = {-0.5f, -0.5f, 0.0f,
                        0.0f, -0.5f, 0.5f,
                        0.5f, -0.5f, 0.0f,
                        0.5f, 0.5f, 0.0f};

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
}

static void CreateIndexBuffer() {
    unsigned indices[] = {0, 3, 1,
                          1, 3, 2,
                          2, 3, 0,
                          0, 1, 2 };

    glGenBuffers(1, &IBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

}

const char *vs_filename = "shaders/tutorial14/shader.vert";
const char *ps_filename = "shaders/tutorial14/shader.frag";

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

    gModelLocation = glGetUniformLocation(shader_program, "model");
    if (gModelLocation == -1) {
        fprintf(stderr, "Error getting uniform location of 'model'\n");
        exit(-1);
    }
    gViewLocation = glGetUniformLocation(shader_program, "view");
    if (gViewLocation == -1) {
        fprintf(stderr, "Error getting uniform location of 'view'\n");
        exit(-1);
    }
    gProjectionLocation = glGetUniformLocation(shader_program, "projection");
    if (gProjectionLocation == -1) {
        fprintf(stderr, "Error getting uniform location of 'projection'\n");
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
    const int win_width = 800;
    const int win_height = 600;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(win_width, win_height);
    glutCreateWindow("Tutorial 15");
    InitializeGlutCallbacks();

    auto res = glewInit();
    if (res != GLEW_OK) {
        fprintf(stderr, "Error: '%s'\n", glewGetErrorString(res));
        return -1;
    }
    if (!glfwInit()) {
        fprintf(stderr, "Error: glfwInit()\n");
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    CreateVertexBuffer();
    CreateIndexBuffer();
    CompileShader();

    glm::mat4 projection = glm::perspective(glm::radians(45.0f), float(win_width) / float(win_height), 0.1f, 100.f);
    glUniformMatrix4fv(gProjectionLocation, 1, GL_FALSE, glm::value_ptr(projection));

    glClearColor(.0f, .0f, .0f, .0f);
    glutMainLoop();

    return 0;
}
