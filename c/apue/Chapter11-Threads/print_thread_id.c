#include <pthread.h>
#include <stdio.h>

void *task1(void *arg) {
  printf("tid1: %lld\n", (long long)pthread_self());
  return NULL;
}

void *task2(void *arg) {
  printf("tid2: %lld\n", (long long)pthread_self());
  return NULL;
}

int main() {
  pthread_t tid1, tid2;
  pthread_create(&tid1, NULL, task1, NULL);
  pthread_create(&tid2, NULL, task2, NULL);

  printf("tid1 == tid2 ? %s\n", pthread_equal(tid1, tid2) ? "true" : "false");
  printf("main thread id: %lld\n", (long long)pthread_self());
  return 0;
}
