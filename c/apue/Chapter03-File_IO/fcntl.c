#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc,char *argv[]) {
	int val;

	if (argc != 2) {
		fprintf(stderr,"Usage: %s <descriptor#1", argv[0]);
		exit(1);
	}
	if ((val = fcntl(atoi(argv[1]),F_GETFL,0)) < 0) {
		fprintf(stderr,"fcntl error for fd %d",atoi(argv[1]));
		exit(1);
	}
	switch (val & O_ACCMODE) {
	case O_RDONLY:
		printf("read only");
		break;
	case O_WRONLY:
		printf("write only");
		break;
	case O_RDWR:
		printf("read write");
		break;
	default:
		fprintf(stderr,"unknow access mode");
		exit(1);
	}

	if (val & O_APPEND) {
		printf(", append");
	}
	if (val & O_NONBLOCK) {
		printf(", nonblocking");
	}
	if (val & O_SYNC) {
		printf(", synchronous writes");
	}
	putchar('\n');
	return 0;
}
