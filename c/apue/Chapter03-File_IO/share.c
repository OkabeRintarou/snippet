#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#define FILENAME "share.file"
#define SIZE 16

void err_sys(const char *msg) {
	perror(msg);
	exit(1);
}

int main() {
	int fd,i;
	char *CHILD_WRITE = (char*)malloc(sizeof(char) * SIZE);
	char *PARENT_WRITE = (char*)malloc(sizeof(char) * SIZE);
	memset(CHILD_WRITE,'a',SIZE);
	memset(PARENT_WRITE,'b',SIZE);

	if ((fd = open(FILENAME,O_RDWR | O_CREAT,0600)) == -1) {
		err_sys("open error");
	}
	if (fork() == 0) {
		for (i = 0; i < 10000;i++) {
			if (lseek(fd,0,SEEK_END) < 0) {
				err_sys("lseek error");
			}
			if (write(fd,CHILD_WRITE,SIZE) != SIZE) {
				err_sys("child write error");
			}
		}
		close(fd);
		exit(0);
	} else {
		for (i = 0; i < 10000;i++) {
			if (lseek(fd,0,SEEK_END) < 0) {
				err_sys("lseek error");
			}
			if (write(fd,PARENT_WRITE,SIZE) != SIZE) {
				err_sys("parent write error");
			}
		}
		close(fd);
	}
	free(CHILD_WRITE);
	free(PARENT_WRITE);
	return 0;
}
