#include <stdio.h>
#include <sys/ptrace.h>
#include <sys/reg.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

int main() {
  pid_t child;
  long orig_eax, eax;
  int status;
  int insyscall = 0;
  struct user_regs_struct regs;

  child = fork();
  if (child == 0) {
    ptrace(PTRACE_TRACEME, 0, NULL, NULL);
    execl("/bin/ls", "/bin/ls", NULL);
  } else {
    for (;;) {
      wait(&status);
      if (WIFEXITED(status)) {
        printf("child process has exited!\n");
        break;
      }
      orig_eax = ptrace(PTRACE_PEEKUSER, child,
                        sizeof(unsigned long) * ORIG_RAX, NULL);
      if (orig_eax == SYS_write) {
        if (insyscall == 0) {
          /* Syscall entry */
          insyscall = 1;
          // %rax			System_call			%rdi								%rsi
          // %rdx 1				sys_write
          // unsigned int fd			const char *buf			size_t
          // count
          ptrace(PTRACE_GETREGS, child, NULL, &regs);
          printf("Write called with %lld, %lld, %lld\n", regs.rdi, regs.rsi,
                 regs.rdx);
        } else {
          /* Syscall exit */
          eax =
              ptrace(PTRACE_PEEKUSER, child, sizeof(unsigned long) * RAX, NULL);
          printf("Write returned with %ld\n", eax);
          insyscall = 0;
        }
      }

      ptrace(PTRACE_SYSCALL, child, NULL, NULL);
    }
  }
  return 0;
}
