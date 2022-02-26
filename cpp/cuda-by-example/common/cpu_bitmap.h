#pragma once

#include <cstdlib>
#include "gl_helper.h"

struct CPUBitmap {
public:
  CPUBitmap(int width, int height, void *d = nullptr) {
    pixels_ = new unsigned char[width * height * 4];
    x_ = width;
    y_ = height;
    data_block_ = d;
  }

  ~CPUBitmap() { delete[] pixels_; }

  unsigned char *get_ptr() const { return pixels_; }
  long image_size() const { return x_ * y_ * 4; }

  void display_and_exit(void (*e)(void *) = nullptr) {
    CPUBitmap **bitmap = get_bitmap_ptr();
    *bitmap = this;
    bitmap_exit_ = e;
    int c = 1;
    char *dummy = const_cast<char*>("");
    glutInit(&c, &dummy);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
    glutInitWindowSize(x_, y_);
    glutCreateWindow("cuda-by-example");
    glutKeyboardFunc(key_func);
    glutDisplayFunc(display_func);
    glutMainLoop();
  }

  static CPUBitmap **get_bitmap_ptr() {
    static CPUBitmap *s_bitmap;
    return &s_bitmap;
  }

  static void key_func(unsigned char key, int x, int y) {
    switch (key) {
    case 27: {
      CPUBitmap *bitmap = *(get_bitmap_ptr());
      if (bitmap->data_block_ && bitmap->bitmap_exit_) {
        bitmap->bitmap_exit_(bitmap->data_block_);
        exit(0);
      }
    }
    }
  }

  static void display_func() {
    CPUBitmap *bitmap = *(get_bitmap_ptr());
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(bitmap->x_, bitmap->y_, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels_);
    glFlush();
  }

private:
  unsigned char *pixels_;
  int x_, y_;
  void *data_block_;
  void (*bitmap_exit_)(void *);
};