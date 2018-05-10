#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct passwd *mygetpwnam(const char *name) {
  struct passwd *ptr;

  setpwent();
  while ((ptr = getpwent()) != NULL) {
    if (strcmp(ptr->pw_name, name) == 0) {
      break;
    }
  }
  endpwent();
  return ptr;
}

int main() {

  struct passwd *ptr;
  if ((ptr = mygetpwnam("syl")) == NULL) {
    printf("can't find user named syl\n");
  } else {
    printf("syl: %d\n", ptr->pw_uid);
  }
  return 0;
}
