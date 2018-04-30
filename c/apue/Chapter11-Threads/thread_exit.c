#include <pthread.h>
#include <stdio.h>

struct foo {
  int a, b, c, d;
};

void *fn(void *arg) { pthread_exit((void *)2); }

void printfoo(const char *s, const struct foo *fp) {
  printf("%s", s);
  printf(" structure at 0x%lx\n", (unsigned long)fp);
  printf("  foo.a = %d\n", fp->a);
  printf("  foo.b = %d\n", fp->b);
  printf("  foo.c = %d\n", fp->c);
  printf("  foo.d = %d\n", fp->d);
}

void *fn2(void *arg) {
  struct foo f = {1, 2, 3, 4};
  printfoo("fn2:\n", &f);
  pthread_exit((void *)&f); /* NOTE: 局部变量在线程退出后内存被回收利用 */
}

int main() {
  pthread_t tid;
  void *tret;
  pthread_create(&tid, NULL, fn, NULL);
  pthread_join(tid, &tret);
  printf("thread exit code %ld\n", (long)tret);

  {
    pthread_t tid2;
    struct foo *fp;
    pthread_create(&tid2, NULL, fn2, NULL);
    pthread_join(tid2, (void **)&fp); /* 读取到垃圾数据 */
    printfoo("main:\n", fp);
  }
  return 0;
}
