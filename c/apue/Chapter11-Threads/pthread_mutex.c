#include <malloc.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NHASH 29
#define HASH(id) (((unsigned long)id) % NHASH)

struct foo *fh[NHASH];

pthread_mutex_t hashlock = PTHREAD_MUTEX_INITIALIZER;

struct foo {
  int f_count;
  pthread_mutex_t f_lock;
  int f_id;
  struct foo *f_next; /* protected by hashlock */
};

struct foo *foo_alloc(int id) {
  struct foo *fp;
  int idx;

  if ((fp = malloc(sizeof(struct foo))) != NULL) {
    fp->f_count = 1;
    fp->f_id = id;
    if (pthread_mutex_init(&fp->f_lock, NULL) != 0) {
      free(fp);
      return NULL;
    }
    idx = HASH(id);
    pthread_mutex_lock(&hashlock);
    fp->f_next = fh[idx];
    fh[idx] = fp;
    pthread_mutex_lock(&fp->f_lock);
    pthread_mutex_unlock(&hashlock);
    /* continue initialization */
    pthread_mutex_unlock(&fp->f_lock);
  }
  return fp;
}

void foo_hold(struct foo *fp) {
  pthread_mutex_lock(&hashlock);
  fp->f_count++;
  pthread_mutex_unlock(&hashlock);
}

struct foo *foo_find(int id) {
  struct foo *fp;
  pthread_mutex_lock(&hashlock);
  for (fp = fh[HASH(id)]; fp != NULL; fp = fp->f_next) {
    if (fp->f_id == id) {
      fp->f_count++;
      break;
    }
  }
  pthread_mutex_unlock(&hashlock);
  return fp;
}

void foo_rele(struct foo *fp) {
  struct foo *tfp;
  int idx;

  pthread_mutex_lock(&hashlock);
  if (--fp->f_count == 0) {
    idx = HASH(fp->f_id);
    tfp = fh[idx];
    if (tfp == fp) {
      fh[idx] = fp->f_next;
    } else {
      while (tfp->f_next != fp) {
        tfp = tfp->f_next;
      }
      tfp->f_next = fp->f_next;
    }
    pthread_mutex_unlock(&hashlock);
    pthread_mutex_unlock(&fp->f_lock);
    free(fp);
  }
  pthread_mutex_unlock(&hashlock);
}

void *even(void *arg) {
  for (int i = 0; i < 100; i += 2) {
    foo_alloc(i);
  }
  return NULL;
}

void *odd(void *arg) {
  for (int i = 1; i < 100; i += 2) {
    foo_alloc(i);
  }
}

int main() {
  pthread_t tid1, tid2;
  pthread_create(&tid1, NULL, odd, NULL);
  pthread_create(&tid2, NULL, even, NULL);
  pthread_join(tid1, NULL);
  pthread_join(tid2, NULL);

  struct foo *fp;

  fp = foo_find(55);
  if (fp) {
    printf("Find: %d,f_count: %d\n", fp->f_id, fp->f_count);
  } else {
    printf("Not found!\n");
  }
  return 0;
}
