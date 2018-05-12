#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int globvar = 6;
char buf[] = "a write to stdout\n";

int main() {
  int var;
  pid_t pid;

  var = 88;
  if (write(STDOUT_FILENO, buf, sizeof(buf) - 1) != sizeof(buf) - 1) {
    perror("write error");
    exit(1);
  }
  printf("before vfork\n"); /* we don't flush stdout */

  if ((pid = vfork()) < 0) {
    perror("fork error");
    exit(1);
  } else if (pid == 0) {
    globvar++;
    var++;
    _exit(0);
  }
  printf("pid = %ld, glob = %d, var = %d\n", (long)getpid(), globvar, var);

  return 0;
}
