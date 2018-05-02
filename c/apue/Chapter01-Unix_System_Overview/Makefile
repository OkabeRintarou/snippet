CC=gcc

TARGETS=ls pid errno uid

all:$(TARGETS)

ls:ls.c
	$(CC) -o $@ $<	
pid:pid.c
	$(CC) -o $@ $<

errno:errno.c
	$(CC) -o $@ $<

uid:uid.c
	$(CC) -o $@ $<

.PHONY:clean
clean:
	-rm $(TARGETS)
