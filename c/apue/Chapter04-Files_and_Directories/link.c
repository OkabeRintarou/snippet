#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
  if (open("tempfile", O_RDWR) < 0) {
    perror("open error");
    exit(1);
  }
  if (unlink("tempfile") < 0) {
    perror("unlink error");
    exit(1);
  }
  printf("file unlinked\n");
  sleep(15);
  printf("done\n");
  return 0;
}
