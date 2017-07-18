#ifndef UTIL_H
#define UTIL_H

#include <cstdio>
#include "opengl.h"

const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 480;
const int SCREEN_FPS = 60;

bool initGL();
void update();
void render();

#endif
