#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/reg.h>
#include <sys/user.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

const int long_size = sizeof(long);

int main() {

  pid_t child;
  child = fork();

  if (child == 0) {
    ptrace(PTRACE_TRACEME, 0, NULL, NULL);
    execl("./dummy", "dummy", NULL);
  } else {

		struct user_regs_struct regs;
		int start = 0;
		long ins;
		int status;

    for (;;) {
      wait(&status);
      if (WIFEXITED(status)) {
        break;
      }

			ptrace(PTRACE_GETREGS, child, NULL, &regs);
			
			if (start == 1) {
				ins = ptrace(PTRACE_PEEKTEXT, child, regs.rip, NULL);
				printf("EIP: %llx Instruction executed: %lx\n", regs.rip, ins);
			} 

			if (regs.orig_rax == SYS_write) {
				start = 1;
				ptrace(PTRACE_SINGLESTEP, child, NULL, NULL);
			} else {
				ptrace(PTRACE_SYSCALL, child, NULL, NULL);
			}
    }
  }
  return 0;
}
