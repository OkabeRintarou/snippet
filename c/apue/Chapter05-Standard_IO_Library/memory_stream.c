#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BSZ 48

int main() {
  FILE *fp;
  char buf[BSZ];

  memset(buf, 'a', BSZ - 2);
  buf[BSZ - 2] = '\0';
  buf[BSZ - 1] = 'X';
  if ((fp = fmemopen(buf, BSZ, "w+")) == NULL) {
    perror("fmemopen error");
    exit(1);
  }
  printf("initial buffer contents: %s\n", buf);
  fprintf(fp, "Hello,World");
  printf("before flush: %s\n", buf);
  fflush(fp); /* a null byte is written when call fflush */
  printf("after fflush: %s\n", buf);
  printf("len of string in buf = %ld\n", (long)strlen(buf));

  memset(buf, 'b', BSZ - 2);
  buf[BSZ - 2] = '\0';
  buf[BSZ - 1] = 'X';
  fprintf(fp, "Hello,World");
  fseek(fp, 0, SEEK_SET); /* a null byte is writeen when call fseek */
  printf("after fseek: %s\n", buf);
  printf("len of string in buf = %ld\n", (long)strlen(buf));

  memset(buf, 'c', BSZ - 2);
  buf[BSZ - 2] = '\0';
  buf[BSZ - 1] = 'X';
  fprintf(fp, "Hello,World");
  fclose(fp);
  printf("after fclose: %s\n", buf);
  printf("len of string in buf = %ld\n", (long)strlen(buf));
  return 0;
}
