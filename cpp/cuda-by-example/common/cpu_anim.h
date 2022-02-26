#pragma once

#include "gl_helper.h"
#include <cstdlib>

struct CPUAnimBitmap {
public:
  CPUAnimBitmap(int w, int h, void *d = nullptr) {
    width_ = w;
    height_ = h;
    pixels_ = new unsigned char[w * h * 4];
    data_block_ = d;
  }

  ~CPUAnimBitmap() { delete[] pixels_; }

  unsigned char *get_ptr() const { return pixels_; }
  long image_size() const { return width_ * height_ * 4; }

  void click_drag(void (*f)(void *, int, int, int, int)) { click_grad_ = f; }

  void anim_and_exit(void (*f)(void *, int), void (*e)(void *)) {
    CPUAnimBitmap **bitmap = get_bitmap_ptr();
    *bitmap = this;
    func_anim_ = f;
    anim_exit_ = e;
    int c = 1;
    char *dummy = const_cast<char *>("");
    glutInit(&c, &dummy);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(width_, height_);
    glutCreateWindow("cuda-by-example");
    glutKeyboardFunc(key_func);
    glutDisplayFunc(display_func);
    if (click_grad_ != nullptr) {
      glutMouseFunc(mouse_func);
    }
    glutIdleFunc(idle_func);
    glutMainLoop();
  }

  static CPUAnimBitmap **get_bitmap_ptr() {
    static CPUAnimBitmap *s_bitmap;
    return &s_bitmap;
  }

  static void mouse_func(int button, int state, int mx, int my) {
    if (button == GLUT_LEFT_BUTTON) {
      CPUAnimBitmap *bitmap = *(get_bitmap_ptr());
      if (state == GLUT_DOWN) {
        bitmap->drag_start_x_ = mx;
        bitmap->drag_start_y_ = my;
      } else if (state == GLUT_UP) {
        bitmap->click_grad_(bitmap->data_block_, bitmap->drag_start_x_,
                            bitmap->drag_start_y_, mx, my);
      }
    }
  }

  static void idle_func() {
    static int ticks = 1;
    CPUAnimBitmap *bitmap = *(get_bitmap_ptr());
    bitmap->func_anim_(bitmap->data_block_, ticks++);
    glutPostRedisplay();
  }

  static void key_func(unsigned char key, int x, int y) {
    if (key == 27) {
      CPUAnimBitmap *bitmap = *(get_bitmap_ptr());
      bitmap->anim_exit_(bitmap->data_block_);
      exit(0);
    }
  }

  static void display_func() {
    CPUAnimBitmap *bitmap = *(get_bitmap_ptr());
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(bitmap->width_, bitmap->height_, GL_RGBA, GL_UNSIGNED_BYTE,
                 bitmap->pixels_);
    glutSwapBuffers();
  }

private:
  unsigned char *pixels_;
  int width_, height_;
  void *data_block_;
  void (*func_anim_)(void *, int) = nullptr;
  void (*anim_exit_)(void *) = nullptr;
  void (*click_grad_)(void *, int, int, int, int) = nullptr;
  int drag_start_x_, drag_start_y_;
};
