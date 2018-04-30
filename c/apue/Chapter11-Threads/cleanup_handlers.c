#include <pthread.h>
#include <stdio.h>

void cleanup(void *arg) { printf("clean up: %s\n", (const char *)arg); }

void *fn1(void *arg) {
  printf("thread 1 start\n");
  pthread_cleanup_push(cleanup, "thread 1 first handler");
  pthread_cleanup_push(cleanup, "thread 1 second handler");
  printf("thread 1 push complete\n");
  if (arg) {
    return (void *)1; /* 通过ret返回时不会自动调用cleanup函数 */
  }
  pthread_cleanup_pop(0);
  pthread_cleanup_pop(0);
  return (void *)1;
}

void *fn2(void *arg) {
  printf("thread 2 start\n");
  pthread_cleanup_push(cleanup, "thread 2 first handler");
  pthread_cleanup_push(cleanup, "thread 2 second handler");
  printf("thread 2 push complete\n");
  if (arg) {
    pthread_exit((void *)2);
  }
  pthread_cleanup_pop(0);
  pthread_cleanup_pop(0);
  pthread_exit((void *)2);
}

/*
** 在Single UNIX
*Specification中,在一对匹配的pthread_cleanup_push和pthread_cleanup_pop中返回时是未定义的行为,
** 这是因为在FreeBSD或Mac OS X中,
*pthread_cleanup_push实现为宏并且在栈中存储一些上下文,如果pthread_cleanup_pop
** 被调用,这些栈中的数据将被 return 错误使用从而可能引发崩溃
*/

int main() {
  pthread_t tid1, tid2;
  int *tret;
  pthread_create(&tid1, NULL, fn1, (void *)1);
  pthread_create(&tid2, NULL, fn2, (void *)1);
  pthread_join(tid1, (void **)&tret);
  printf("thread1 exit code: %ld\n", (long)tret);
  pthread_join(tid2, (void **)&tret);
  printf("thread2 exit code: %ld\n", (long)tret);
  return 0;
}
