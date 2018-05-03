#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
#ifdef BUFFSIZE
	int n;
	char buf[BUFFSIZE];

	while ((n = read(STDIN_FILENO,buf,BUFFSIZE)) > 0) {
		if (write(STDOUT_FILENO,buf,n) != n) {
			perror("write error");
		}
	}
	if (n < 0) {
		perror("read error");
	}
#else
#error "BUFFSIZE not defined"
#endif
	return 0;
}
