#include <ctime>
#include <cstdlib>

void generateNumbers(double *ptr, int size) {
  for (int i = 0; i < size; i++) {
    ptr[i] = double(rand() % 10);
  }
}