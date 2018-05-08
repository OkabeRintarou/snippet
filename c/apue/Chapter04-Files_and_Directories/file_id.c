#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <create-filename>", argv[0]);
    exit(1);
  }

  if (creat(argv[1], 0600) < 0) {
    perror("creat error");
    exit(1);
  }
  struct stat buf;
  if (lstat(argv[1], &buf) < 0) {
    perror("lstat error");
    exit(1);
  }
  printf("user id of owner: %d\n", buf.st_uid);
  printf("group id of owner: %d\n", buf.st_gid);
  return 0;
}
