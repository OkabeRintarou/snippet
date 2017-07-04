CC = gcc
CFLAGS = -std=c99 -Wall -g
BIN = id printenv longjmp
.PHONY: all

SRCS = $(wildcard *.c)
all:$(BIN)

id:id.c
	$(CC) $(CFLAGS) -o $@ $< 
printenv:printenv.c
	$(CC) $(CFLAGS) -o $@ $< 
longjmp:longjmp.c
	$(CC) $(CFLAGS) -o $@ $< 

.PHONY:clean
clean:
	-rm $(BIN)
