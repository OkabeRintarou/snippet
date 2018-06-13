#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <linux/if_tun.h>
#include <stdlib.h>
#include <stdio.h>

int alloc_tun(int flags) {
	struct ifreq ifr;
	int fd,err;
	const  char *dev = "/dev/net/tun";

	if ((fd = open(dev,O_RDWR)) < 0) {
		return fd;
	}

	memset(&ifr,0,sizeof(ifr));
	ifr.ifr_flags = flags;

	if ((err = ioctl(fd,TUNSETIFF,(void*)&ifr)) < 0) {
		close(fd);
		return err;
	}

	printf("Open tun/tap device: %s for reading...\n",ifr.ifr_name);

	return fd;
}

int main() {
	int tun_fd,nread;
	char buffer[1500];

	tun_fd = alloc_tun(IFF_TUN | IFF_NO_PI);

	if (tun_fd < 0) {
		perror("Allocating interface");
		exit(1);
	}

	for(;;) {
		nread = read(tun_fd,buffer,sizeof(buffer));
		if (nread < 0) {
			perror("Reading from interface");
			close(tun_fd);
			exit(1);
		}
		printf("Read %d bytes from tun/tap device\n",nread);
	}
	return 0;
}
