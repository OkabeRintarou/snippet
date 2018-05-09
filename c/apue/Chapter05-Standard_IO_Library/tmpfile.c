#include <stdio.h>
#include <stdlib.h>

#define MAXLINE 8192

int main() {
  char name[L_tmpnam], line[MAXLINE];
  FILE *fp;

  printf("%s\n", tmpnam(NULL));

  tmpnam(name);
  printf("%s\n", name);

  if ((fp = tmpfile()) == NULL) {
    perror("tmpfile error");
    exit(1);
  }
  fputs("one line of output\n", fp);
  rewind(fp);
  if (fgets(line, sizeof(line), fp) == NULL) {
    perror("fgets error");
    exit(1);
  }
  fputs(line, stdout);
  return 0;
}
