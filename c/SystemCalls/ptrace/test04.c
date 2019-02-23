#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/reg.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

const int long_size = sizeof(long);

void reverse(char *str) {
  int i, j;
  char t;

  for (i = 0, j = strlen(str) - 2; i <= j; ++i, --j) {
    t = str[i];
    str[i] = str[j];
    str[j] = t;
  }
}

void getdata(pid_t child, long addr, char *str, int len) {
  char *laddr;
  int i, j;
  union u {
    long val;
    char chars[long_size];
  } data;

  i = 0;
  j = len / long_size;
  laddr = str;

  while (i < j) {
    data.val = ptrace(PTRACE_PEEKDATA, child, addr + i * long_size, NULL);
    memcpy(laddr, data.chars, long_size);
    ++i;
    laddr += long_size;
  }

  j = len % long_size;
  if (j != 0) {
    data.val = ptrace(PTRACE_PEEKDATA, child, addr + i * long_size, NULL);
    memcpy(laddr, data.chars, j);
  }
  str[len] = '\0';
}

void putdata(pid_t child, long addr, char *str, int len) {
  char *laddr;
  int i, j;
  union u {
    long val;
    char chars[long_size];
  } data;

  i = 0;
  j = len / long_size;
  laddr = str;

  while (i < j) {
    memcpy(data.chars, laddr, long_size);
    ptrace(PTRACE_POKEDATA, child, addr + i * long_size, data.val);
    ++i;
    laddr += long_size;
  }

  j = len % long_size;
  if (j != 0) {
    memcpy(data.chars, laddr, j);
    ptrace(PTRACE_POKEDATA, child, addr + i * long_size, data.val);
  }
}

int main() {

  pid_t child;
  child = fork();

  if (child == 0) {
    ptrace(PTRACE_TRACEME, 0, NULL, NULL);
    execl("/bin/ls", "ls", NULL);
  } else {
    long orig_eax;
    long params[3];
    int status;
    char *str, *laddr;
    int toggle = 0;

    for (;;) {
      wait(&status);

      if (WIFEXITED(status)) {
        break;
      }
      orig_eax = ptrace(PTRACE_PEEKUSER, child,
                        sizeof(unsigned long) * ORIG_RAX, NULL);

      if (orig_eax == SYS_write) {
        if (toggle == 0) {
          toggle = 1;
          // %rax			System_call			%rdi								%rsi
          // %rdx 1				sys_write
          // unsigned int fd			const char *buf			size_t
          // count
          params[0] =
              ptrace(PTRACE_PEEKUSER, child, sizeof(unsigned long) * RDI, NULL);
          params[1] =
              ptrace(PTRACE_PEEKUSER, child, sizeof(unsigned long) * RSI, NULL);
          params[2] =
              ptrace(PTRACE_PEEKUSER, child, sizeof(unsigned long) * RDX, NULL);

          str = (char *)calloc((params[2] + 1), sizeof(char));
          getdata(child, params[1], str, params[2]);
          reverse(str);
          putdata(child, params[1], str, params[2]);
        } else {
          toggle = 0;
        }
      }

      ptrace(PTRACE_SYSCALL, child, NULL, NULL);
    }
  }
  return 0;
}
