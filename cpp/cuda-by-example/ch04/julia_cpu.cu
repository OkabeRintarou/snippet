#include "cpu_bitmap.h"

static const unsigned DIM = 1000;

struct cuComplex {
  float r;
  float i;
  cuComplex(float a, float b) : r(a), i(b) {}
  float magnitude2() const { return r * r + i * i; }
  cuComplex operator*(const cuComplex &a) {
    return {r * a.r - i * a.i, i * a.r + r * a.i};
  }
  cuComplex operator+(const cuComplex &a) { return {r + a.r, i + a.i}; }
};

int julia(int x, int y) {
  const float scale = 1.5f;
  const auto dim = static_cast<float>(DIM);
  float jx = scale * (dim / 2.f - float(x)) / (dim / 2.f);
  float jy = scale * (dim / 2.f - float(y)) / (dim / 2.f);

  cuComplex c(-0.8f, 0.156f);
  cuComplex a(jx, jy);

  for (int i = 0; i < 200; i++) {
    a = a * a + c;
    if (a.magnitude2() > 1000.f) {
      return 0;
    }
  }
  return 1;
}
static void kernel(unsigned char *ptr) {
  for (int y = 0; y < DIM; y++) {
    for (int x = 0; x < DIM; x++) {
      int offset = y * DIM + x;

      int julia_value = julia(x, y);
      ptr[offset * 4 + 0] = 255 * julia_value;
      ptr[offset * 4 + 1] = 0;
      ptr[offset * 4 + 2] = 0;
      ptr[offset * 4 + 3] = 255;
    }
  }
}

int main() {
  CPUBitmap bitmap(DIM, DIM);
  unsigned char *ptr = bitmap.get_ptr();
  kernel(ptr);
  bitmap.display_and_exit();
  return 0;
}
