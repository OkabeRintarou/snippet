#include <malloc.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

struct msg {
  struct msg *next;
  int id;
};

struct msg *workq;

pthread_cond_t qready = PTHREAD_COND_INITIALIZER;

pthread_mutex_t qlock = PTHREAD_MUTEX_INITIALIZER;

void process_msg(void) {
  struct msg *mp;

  for (;;) {
    pthread_mutex_lock(&qlock);

    while (workq == NULL) {
      pthread_cond_wait(&qready, &qlock);
    }

    mp = workq;
    workq = workq->next;
    pthread_mutex_unlock(&qlock);

    if (mp->id < 0) {
      break;
    } else {
      printf("Thread %lx process id : %d\n", (unsigned long)pthread_self(),
             mp->id);
    }
    usleep((rand() % 15) * 1000);
  }
}

void enqueue_msg(struct msg *mp) {
  pthread_mutex_lock(&qlock);
  mp->next = workq;
  workq = mp;
  pthread_mutex_unlock(&qlock);
  pthread_cond_signal(&qready);
}

void *consumer(void *arg) { process_msg(); }

void *producer(void *arg) {
  int i;
  struct msg *mp;
  for (i = 0; i < 100; i++) {
    mp = (struct msg *)malloc(sizeof(struct msg));
    mp->next = NULL;
    mp->id = i;
    printf("Thread %lx produce message %d\n", (unsigned long)pthread_self(),
           mp->id);
    enqueue_msg(mp);

    usleep(((rand() % 8) + 2) * 1000);
  }
  mp = (struct msg *)malloc(sizeof(struct msg));
  mp->next = NULL;
  mp->id = -1;
  printf("Thread %lx produce message %d\n", (unsigned long)pthread_self(),
         mp->id);
  enqueue_msg(mp);
}

int main() {
  pthread_t c, p;
  pthread_create(&c, NULL, consumer, NULL);
  pthread_create(&p, NULL, producer, NULL);
  pthread_join(c, NULL);
  pthread_join(p, NULL);
  return 0;
}
