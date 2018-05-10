#include <shadow.h>
#include <stdio.h>

int main() {
  struct spwd *ptr;
  if ((ptr = getspnam("syl")) == NULL) {
    printf("can't query shadow name of syl\n");
  } else {
    printf("syl: %s\n", ptr->sp_pwdp);
  }
  return 0;
}
