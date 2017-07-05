CC = gcc
SRCS = `ls *.c`
BINS = $(SRCS:.c= )
CFLAGS = -Wall -g

all: passwd group scan crypt

passwd:passwd.c
	$(CC) $(CFLAGS) -o $@ $<
group:group.c
	$(CC) $(CFLAGS) -o $@ $<
scan:scan.c
	$(CC) $(CFLAGS) -o $@ $<
crypt:crypt.c
	$(CC) $(CFLAGS) -o $@ $< -lcrypt

.PHONY:clean
clean:
	-rm passwd group scan crypt
