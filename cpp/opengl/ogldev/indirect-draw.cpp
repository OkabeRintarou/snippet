#include "util.h"
/* clang-format off */
#include <GL/glew.h>
#include <GL/freeglut.h>
/* clang-format on */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <glm/glm.hpp>
#include <string>

GLuint VBO;
GLuint IBO;
GLuint INDIRECT_BO;
GLint gTranslationLocation;

static void RenderScreenCB() {
    static float scale = 0.0f;
    static float delta = 0.01f;

    scale += delta;
    glm::mat4 mat(cosf(scale), 0.0f, -sinf(scale), 0.0f,
                  0.0f, 1.0f, 0.0f, 0.0f,
                  sinf(scale), 0.0f, cosf(scale), 0.0f,
                  0.0f, 0.0f, 0.0f, 1.0f);

    glUniformMatrix4fv(gTranslationLocation, 1, GL_TRUE, &mat[0][0]);

    glClear(GL_COLOR_BUFFER_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *) 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, INDIRECT_BO);

    glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, (void *) 0);

    glDisableVertexAttribArray(0);

    glutPostRedisplay();
    glutSwapBuffers();
}

static void InitializeGlutCallbacks() {
    glutDisplayFunc(RenderScreenCB);
    glutIdleFunc(RenderScreenCB);
}

static void CreateVertexBuffer() {
    float vertices[] = {-1.0f, -1.0f, 0.0f,
                        0.0f, -1.0f, 1.0f,
                        1.0f, -1.0f, 0.0f,
                        0.0f, 1.0f, 0.0f};

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
}

static void CreateIndexBuffer() {
    unsigned indices[] = {0, 3, 1,
                          1, 3, 2,
                          2, 3, 0,
                          0, 1, 2};

    glGenBuffers(1, &IBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

struct DrawElementsIndirectCommand {
    uint32_t count;
    uint32_t instance_count;
    uint32_t first_index;
    uint32_t base_vertex;
    uint32_t base_instance;
};

static void CreateIndirectDrawBuffer() {
    glGenBuffers(1, &INDIRECT_BO);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, INDIRECT_BO);

    DrawElementsIndirectCommand command{0};
    command.count = 12;
    command.instance_count = 1;
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(command), &command, GL_STATIC_DRAW);
}

const char *vs_filename = "shaders/tutorial09/shader.vert";
const char *ps_filename = "shaders/tutorial09/shader.frag";

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

    gTranslationLocation = glGetUniformLocation(shader_program, "gTranslation");
    if (gTranslationLocation == -1) {
        fprintf(stderr, "Error getting uniform location of 'gTranslation'\n");
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
    glutCreateWindow("Indirect Draw Example");
    InitializeGlutCallbacks();

    auto res = glewInit();
    if (res != GLEW_OK) {
        fprintf(stderr, "Error: '%s'\n", glewGetErrorString(res));
        return -1;
    }

    CreateVertexBuffer();
    CreateIndexBuffer();
    CreateIndirectDrawBuffer();
    CompileShader();

    glClearColor(.0f, .0f, .0f, .0f);
    glutMainLoop();
    return 0;
}
