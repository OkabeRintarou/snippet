#include <stdio.h>
#include <string.h>
#include <errno.h>

int main() {
	fprintf(stderr,"EACCES: %s\n",strerror(EACCES));
	errno = ENOENT;
	perror("ERR");
	return 0;
}
