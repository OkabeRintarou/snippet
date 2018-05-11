CC=gcc

TARGETS=hello_89 hello_99 atexit args setjmp rlimit

all:$(TARGETS)

hello_89:hello.c
	$(CC) -std=c89 -o $@ $<

hello_99:hello.c
	$(CC) -std=c99 -o $@ $<

atexit:atexit.c
	$(CC) -o $@ $<

args:args.c
	$(CC) -o $@ $<

setjmp:setjmp.c
	$(CC) -o $@ $<

rlimit:rlimit.c
	$(CC) -o $@ $<

.PHONY:clean

clean:
	-rm $(TARGETS)
