CC=gcc

TARGETS=stdin_seek hole open share fcntl

all:$(TARGETS)

stdin_seek:stdin_seek.c
	$(CC) -o $@ $<

hole:hole.c
	$(CC) -o $@ $<

open:open.c
	$(CC) -o $@ $<

share:share.c
	$(CC) -o $@ $<

fcntl:fcntl.c
	$(CC) -o $@ $<

.PHONY:clean

clean:
	-rm $(TARGETS)
