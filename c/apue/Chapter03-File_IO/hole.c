#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

char buf1[] = "abcdefghij";
char buf2[] = "ABCDEFGHIJ";

void err_sys(const char *msg) {
	perror(msg);
	exit(1);
}

int main() {
	int fd;
	if ((fd = creat("file.hole",0600)) < 0) {
		err_sys("create error");
	}
	if (write(fd,buf1,10) != 10) {
		err_sys("buf1 write error");
	}

	/* offset now 10 */
	if (lseek(fd,16384,SEEK_SET) == -1) {
		err_sys("lseek error");
	}
	/* offset now 16384 */
	if (write(fd,buf2,10) != 10) {
		err_sys("buf2 write error");
	}
	/* offset now 16394 */
	close(fd);
	/* next create a file without hole, but the same size and content */
	if ((fd = creat("file.nohole",0600)) < 0) {
		err_sys("create error");
	}
	if (write(fd,buf1,10) != 10) {
		err_sys("buf1 write error");
	}
	int i;
	for (i = 0; i < 16384 - 10; i++) {
		if (write(fd,"0",1) != 1) {
			err_sys("write 0 error");
		}
	}
	if (write(fd,buf2,10) != 10) {
		err_sys("buf2 write error");
	}
	close(fd);

	return 0;
}
