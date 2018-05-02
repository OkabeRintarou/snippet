#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
	if (fork() == 0) {
		printf("child process id: %lx\n",(unsigned long)getpid());
		exit(0);
	}
	printf("parent process id: %lx\n",(unsigned long)getpid());
	return 0;
}
