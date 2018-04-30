CC=gcc
LINK=-lpthread

TARGETS=print_thread_id thread_exit cleanup_handlers pthread_mutex pthread_mutex_timedlock pthread_cond pthread_barrier
all:$(TARGETS)

print_thread_id:print_thread_id.c
	$(CC) -o $@ $< $(LINK) 

thread_exit:thread_exit.c
	$(CC) -o $@ $< $(LINK) 

cleanup_handlers:cleanup_handlers.c
	$(CC) -o $@ $< $(LINK) 


pthread_mutex:pthread_mutex.c
	$(CC) -o $@ $< $(LINK) 

pthread_mutex_timedlock:pthread_mutex_timedlock.c
	$(CC) -o $@ $< $(LINK) 

pthread_cond:pthread_cond.c
	$(CC) -o $@ $< $(LINK) 

pthread_barrier:pthread_barrier.c
	$(CC) -o $@ $< $(LINK)

.PHONY:clean
clean:
	-rm $(TARGETS)
