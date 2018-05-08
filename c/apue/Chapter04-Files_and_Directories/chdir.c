#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
  DIR *dp;
  struct dirent *dirp;

  if (chdir("/tmp") < 0) {
    perror("chidir error");
    exit(1);
  }
  if ((dp = opendir(".")) == NULL) {
    perror("opendir error");
    exit(1);
  }
  while ((dirp = readdir(dp)) != NULL) {
    printf("%s\n", dirp->d_name);
  }
  return 0;
}
