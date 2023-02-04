#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cstdio>

GLuint VBO;

static void RenderScreenCB() {
   glClear(GL_COLOR_BUFFER_BIT);
   glBindBuffer(GL_ARRAY_BUFFER, VBO);
   glEnableVertexAttribArray(0);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glDisableVertexAttribArray(0);

   glutSwapBuffers();
}

static void CreateVertexBuffer() {
    float vertices[] = {0.5f, 0.5f, 0.0f,
                            0.5f, -0.5f, 0.0f,
                            -0.5f, -0.5f, 0.0f};

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
}

int main(int argc, char *argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(1024, 768);
    glutCreateWindow("Tutorial 01");
    glutDisplayFunc(RenderScreenCB);

    auto res = glewInit();
    if (res != GLEW_OK) {
        fprintf(stderr, "Error: '%s'\n", glewGetErrorString(res));
        return -1;
    }

    CreateVertexBuffer();

    glClearColor(.0f, .0f, .0f, .0f);
    glutMainLoop();
    return 0;
}
