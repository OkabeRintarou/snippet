#include <GL/freeglut.h>

static void RenderScreenCB() {
   glClear(GL_COLOR_BUFFER_BIT);
   glutSwapBuffers();
}

int main(int argc, char *argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(1024, 768);
    glutCreateWindow("Tutorial 01");
    glutDisplayFunc(RenderScreenCB);

    glClearColor(.0f, .0f, .0f, .0f);
    glutMainLoop();
    return 0;
}
