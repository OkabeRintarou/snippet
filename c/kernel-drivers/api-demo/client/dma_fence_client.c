#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <pthread.h>

#define DMA_FENCE_IN_CMD 		_IOWR('f', 0, int)
#define DMA_FENCE_OUT_CMD		_IOWR('f', 1, int)
#define DMA_FENCE_SIGNAL_CMD	_IO('f', 2)

#define BLOCKING_IN_KERNEL

int fd = -1;

static int sync_wait(int fd, int timeout) {
	struct pollfd fds = {0};
	int ret;

	fds.fd = fd;
	fds.events = POLLIN;

	do {
		ret = poll(&fds, 1, timeout);
		if (ret < 0) {
			if (fds.revents & (POLLERR | POLLNVAL)) {
				errno = EINVAL;
				return -1;
			}
			return 0;
		} else if (ret == 0) {
			errno = ETIME;
			return -1;
		}
	} while (ret == -1 && (errno == EINTR || errno == EAGAIN));

	return ret;
}

static void *signal_thread(void *arg) {
	sleep(5);

	if (ioctl(fd, DMA_FENCE_SIGNAL_CMD) < 0) {
		perror("Signal out fence fd failed\n");
	}

	return NULL;
}

int main() {
	int out_fence_fd;
	pthread_t tid;

	fd = open("/dev/dma-fence", O_RDWR | O_NONBLOCK, 0);
	if (fd < 0) {
		fprintf(stderr, "Cannot open dma-fence dev: %s\n", strerror(errno));
		exit(1);
	}

	if (ioctl(fd, DMA_FENCE_OUT_CMD, &out_fence_fd) < 0) {
		perror("Get out fence fd failed\n");
		close(fd);
		return -1;
	}

	printf("Get an out-fence fd: %d\n", out_fence_fd);

	if (pthread_create(&tid, NULL, signal_thread, NULL) < 0) {
		fprintf(stderr, "Create thread failed\n");
		close(out_fence_fd);
		close(fd);
		return -1;
	}

#ifdef BLOCKING_IN_KERNEL
	printf("Waiting out-fence to be signaled on KERNEL side...\n");
	if (ioctl(fd, DMA_FENCE_IN_CMD, &out_fence_fd) < 0) {
		perror("Get out fence fd failed\n");
		close(out_fence_fd);
		close(fd);
		return -1;
	}
#else
	printf("Waiting out-fence to be signaled on USER side...\n");
	sync_wait(fd, -1);
#endif
	
	printf("Out-fence is signaled\n");

	if (pthread_join(tid, NULL)) {
		printf("Thread is not exit...\n");
		return -1;
	}

	close(out_fence_fd);
	close(fd);

	return 0;
}
