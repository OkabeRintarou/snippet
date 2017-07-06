CC = gcc
CFLAGS = -Wall -g

BINS = time

all:$(BINS)

time:time.c
	$(CC) -o $@ $< $(CFLAGS)

.PHONY:clean
clean:
	-rm $(BINS)
