#include <stdio.h>
#include <sys/utsname.h>

int main() {
  struct utsname info;
  if ((uname(&info)) < 0) {
    perror("uname error");
    return 1;
  }
  printf("operating system name : %s\n", info.sysname);
  printf("node name             : %s\n", info.nodename);
  printf("release               : %s\n", info.release);
  printf("version               : %s\n", info.version);
  printf("machine               : %s\n", info.machine);
  return 0;
}
