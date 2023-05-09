#include "util.h"
#include <amdgpu.h>
#include <amdgpu_drm.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static void send_fd(int fd) {
  int sockfd;
  msghdr msg;
  cmsghdr *cmsg;
  char buf[CMSG_SPACE(sizeof(int))], dummy;
  sockaddr_un addr;
  int len, r;

  memset(&msg, 0, sizeof(msg));

  sockfd = socket(AF_UNIX, SOCK_STREAM, 0);

  addr.sun_family = AF_UNIX;
  strcpy(addr.sun_path, SERVER_SOCKET);
  len = sizeof(addr);

  if (connect(sockfd, (sockaddr *)&addr, len) < 0) {
    perror("connect");
    exit(-1);
  }

  memset(buf, 0, sizeof(buf));
  struct iovec io = {.iov_base = &dummy, .iov_len = sizeof(dummy)};

  msg.msg_iov = &io;
  msg.msg_iovlen = 1;
  msg.msg_control = buf;
  msg.msg_controllen = sizeof(buf);

  cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  *(int *)CMSG_DATA(cmsg) = fd;

  if (sendmsg(sockfd, &msg, 0) < 0) {
    perror("sendmsg");
    exit(-1);
  }

  printf("Client send fd %d\n", fd);

  close(sockfd);
}

static amdgpu_bo_handle amdgpu_mem_alloc(amdgpu_device_handle device_handle,
                                         uint64_t size, uint64_t align,
                                         uint32_t type, uint64_t flags) {

  int r;
  amdgpu_bo_alloc_request req{};
  amdgpu_bo_handle buf_handle = nullptr;

  req.alloc_size = size;
  req.phys_alignment = align;
  req.preferred_heap = type;
  req.flags = flags;

  r = amdgpu_bo_alloc(device_handle, &req, &buf_handle);
  if (r != 0) {
    fprintf(stderr, "Fail to alloc amdgpu bo\n");
    return nullptr;
  }

  void *addr = nullptr;
  r = amdgpu_bo_cpu_map(buf_handle, &addr);
  if (r != 0 || addr == nullptr) {
    fprintf(stderr, "Fail to map amdgpu bo\n");
    amdgpu_bo_free(buf_handle);
    return nullptr;
  }

  auto ptr = static_cast<uint32_t *>(addr);
  const uint32_t colors[] = {0xff0000ff, 0xff00ff00, 0xffff0000, 0xffffffff};
  size /= sizeof(uint32_t);
  auto gap = size / 4u;

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < gap; j++) {
      ptr[i * gap + j] = colors[i];
    }
  }
  return buf_handle;
}

int main() {
  int dev_fd = open_device();
  if (dev_fd < 0) {
    fprintf(stderr, "Fail to open device\n");
    exit(-1);
  }

  amdgpu_device_handle dev_handle = nullptr;
  uint32_t major, minor;

  if (amdgpu_device_initialize(dev_fd, &major, &minor, &dev_handle) != 0) {
    fprintf(stderr, "Fail to init amdgpu device\n");
    close(dev_fd);
    exit(-1);
  }

  const uint64_t buf_size = 256 * 256 * 4;
  amdgpu_bo_handle buf_handle =
      amdgpu_mem_alloc(dev_handle, buf_size, 4096, AMDGPU_GEM_DOMAIN_GTT, 0);
  if (!buf_handle) {
    fprintf(stderr, "Fail to alloc amdgpu bo\n");
    exit(-1);
  }

  uint32_t shared_handle = -1;
  if (amdgpu_bo_export(buf_handle, amdgpu_bo_handle_type_dma_buf_fd,
                       &shared_handle) != 0) {
    fprintf(stderr, "Fail to export handle\n");
    exit(-1);
  }
  send_fd(static_cast<int>(shared_handle));
  return 0;
}
