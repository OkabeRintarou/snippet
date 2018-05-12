#include <stdio.h>
#include <unistd.h>

int main() {
  printf("process id        : %d\n", getpid());
  printf("parent process id : %d\n", getppid());
  printf("real user id      : %d\n", getuid());
  printf("effective user id : %d\n", geteuid());
  printf("real group id     : %d\n", getgid());
  printf("effective group id: %d\n", getegid());
  return 0;
}
