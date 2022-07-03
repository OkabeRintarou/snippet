#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#define PAGE_SIZE (4 * 1024)
#define BUF_SIZE (16 * PAGE_SIZE)
#define OFFSET (16 * PAGE_SIZE)

int main(int argc, char *argv[]) {
	int fd;
	char *addr = NULL;

	fd = open("/dev/remap_pfn", O_RDWR);
	if (fd < 0) {
		fprintf(stderr, "failed to open, errno = %s[%d]\n", strerror(errno), errno);
		return -1;
	}
	addr = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_LOCKED, fd, OFFSET); 
	if (addr == MAP_FAILED) {
		fprintf(stderr, "failed to mmap, errno = %s[%d]\n", strerror(errno), errno);
		return -1;
	}

	printf("%s\n", (const char *)addr);

	for (;;) {
		sleep(1);
	}

	munmap(addr, BUF_SIZE);
	close(fd);
	return 0;
}
