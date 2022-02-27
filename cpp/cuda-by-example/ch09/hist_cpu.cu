#include <ctime>
#include <algorithm>
#include "common.h"

static constexpr int SIZE = 100 * 1024 * 1024;

int main() {
  auto buffer = static_cast<unsigned char *>(big_random_block(SIZE));
  clock_t start, stop;
  start = clock();

  unsigned int histo[256];
  std::fill(std::begin(histo), std::end(histo), 0);
  for (int i = 0; i < SIZE; i++) {
    histo[buffer[i]]++;
  }
  stop = clock();
  float elapsed_time = float(stop - start) / float(CLOCKS_PER_SEC) * 1000.0f;
  printf("Time to generate: %3.1f ms\n", elapsed_time);

  long histo_count = 0;
  for (int i = 0; i < 256; i++) {
    histo_count += histo[i];
  }
  printf("Histogram Sum: %ld\n", histo_count);
  delete []buffer;
  return 0;
}
