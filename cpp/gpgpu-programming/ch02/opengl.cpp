#include "file_reader.h"
#include "gl_helper.h"
#include <cstdio>
#include <cstdlib>
#include <string>

void changeSize(int w, int h) {
  if (h == 0) {
    h = 1;
  }
  float ratio = 1.0f * float(w) / float(h);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glViewport(0, 0, w, h);
  gluPerspective(45, ratio, 1, 1000);
  glMatrixMode(GL_MODELVIEW);
}

GLuint v, f, p;
float lpos[4] = {1.0f, 0.5f, 1.0f, 0.0f};
float a = 0.0f;
GLint time_id;

void renderScene(void) {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();
  gluLookAt(0.0, 0.0, 5.0, 0.0, 0.0, -1.0, 0.0f, 1.0f, 0.0f);
  glLightfv(GL_LIGHT0, GL_POSITION, lpos);
  glRotatef(a, 0.0f, 1.0f, 1.0f);
  glutSolidTeapot(1.0);
  a += 0.1f;
  glUniform1f(time_id, a);
  glutSwapBuffers();
}

void setShaders() {
  v = glCreateShader(GL_VERTEX_SHADER);
  f = glCreateShader(GL_FRAGMENT_SHADER);

  std::string vs = file_read("shader/passthrough.vert");
  if (vs.empty()) {
    fprintf(stderr, "Fail to read content from \"passthrough.vert\"");
    exit(1);
  }
  std::string fs = file_read("shader/uniform.frag");
  if (fs.empty()) {
    fprintf(stderr, "Fail to read content from \"uniform.frag\"");
    exit(1);
  }

  const char *vv = vs.c_str();
  const char *ff = fs.c_str();

  glShaderSource(v, 1, &vv, nullptr);
  glShaderSource(f, 1, &ff, nullptr);

  int success;
  char info_log[512];

  glCompileShader(v);
  glGetShaderiv(v, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(v, 512, nullptr, info_log);
    printf("Compile vertex shader failed: %s\n", info_log);
  }

  glCompileShader(f);
  glGetShaderiv(f, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(f, 512, nullptr, info_log);
    printf("Compile fragment shader failed: %s\n", info_log);
  }

  p = glCreateProgram();
  glAttachShader(p, v);
  glAttachShader(p, f);
  glLinkProgram(p);
  glGetProgramiv(p, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(p, 512, nullptr, info_log);
    printf("Link program failed: %s\n", info_log);
  }

  glUseProgram(p);

  time_id = glGetUniformLocation(p, "v_time");
}

int main(int argc, char *argv[]) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(320, 320);
  glutCreateWindow("GPGPU Tutorial");
  glutDisplayFunc(renderScene);
  glutIdleFunc(renderScene);
  glutReshapeFunc(changeSize);
  glEnable(GL_DEPTH_TEST);
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glColor3f(1.0f, 1.0f, 1.0f);

  glewInit();

  setShaders();

  glutMainLoop();
  return 0;
}
