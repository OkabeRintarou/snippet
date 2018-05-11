#include <stdio.h>
#include <stdlib.h>

static void my_atexit1(void);
static void my_atexit2(void);

int main() {
  if (atexit(my_atexit2) != 0) {
    perror("atexit error");
    exit(1);
  }
  if (atexit(my_atexit1) != 0) {
    perror("atexit error");
    exit(1);
  }
  if (atexit(my_atexit1) != 0) {
    perror("atexit error");
    exit(1);
  }
  return 0;
}

static void my_atexit1(void) { printf("first exit handler\n"); }

static void my_atexit2(void) { printf("second exit handler\n"); }
