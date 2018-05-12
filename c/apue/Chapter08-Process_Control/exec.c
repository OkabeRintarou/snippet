#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

char *env_init[] = {"USER=unknow", "PATH=/tmp", NULL};

int main() {
  pid_t pid;

  if ((pid = fork()) < 0) {
    perror("fork error");
    exit(1);
  } else if (pid == 0) {
    if (execle("/bin/echo", "echo", "myarg1", "MYARG2", NULL, env_init) < 0) {
      perror("execle error");
      exit(1);
    }
  }

  if (waitpid(pid, NULL, 0) < 0) {
    perror("waitpid error");
    exit(1);
  }

  if ((pid = fork()) < 0) {
    perror("fork error");
    exit(1);
  } else if (pid == 0) {
    if (execlp("echo", "echo", "myarg1", "MYARG2", NULL) < 0) {
      perror("execlp error");
    }
  }
  return 0;
}
