#include <iostream>
#include <timer.h>
#include <vector>

static const int WIDTH = 10240;
static const int HEIGHT = 10240;
static const int CHANNEL = 4;
static const int RADIUS = 2;

using namespace std;

static inline size_t offset(size_t channel, size_t row, size_t column) {
  return channel * HEIGHT * WIDTH + row * HEIGHT + column;
}

int main() {
  std::vector<float> input(WIDTH * HEIGHT * CHANNEL);
  std::vector<float> output(WIDTH * HEIGHT * CHANNEL, 0);
  float data = .0f;
  for (int k = 0; k < CHANNEL; k++) {
    for (int j = 0; j < HEIGHT; j++) {
      for (int i = 0; i < WIDTH; i++) {
        size_t o= offset(k, j, i);
        input[o] = data;
        data += 1.0f;
      }
    }
  }

  float sum = .0f;
  size_t total = 0;

  Timer timer;
  timer.reset();

  for (int k = 0; k < CHANNEL; k++) {
    for (int j = 0; j < HEIGHT; j++) {
      for (int i = 0; i < WIDTH; i++) {

        for (int ii = i - RADIUS; ii <= i + RADIUS; ii++) {
          for (int jj = j - RADIUS; jj <= j + RADIUS; jj++) {
            if (ii >= 0 && ii < WIDTH && jj >= 0 && jj < HEIGHT) {
              sum += input[offset(k, jj, ii)];
              total++;
            }
          }
        }

        output[offset(k, j,i)] = sum / float(total);
        total = 0;
        sum = 0.0f;
      }
    }
  }

  double elapsed = timer.elapsed_milliseconds();
  std::cout << "Time elapsed: " << elapsed << " ms" << std::endl;

  return 0;
}