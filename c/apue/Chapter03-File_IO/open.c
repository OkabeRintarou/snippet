#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

#define SIZE 257

int main() {
	char name[SIZE];
	int i;
	for (i = 0; i < SIZE-1;i++) {
		name[i] = 'a' + (i % 26);
	}
	name[SIZE-1] = '\0';
	int fd;
	if ((fd = open(name,O_CREAT | O_RDWR)) < 0) {
		perror("open");
	}
	return 0;
}
