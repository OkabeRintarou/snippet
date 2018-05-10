CC=gcc

TARGETS=pwd shadow uname

all:$(TAGRETS)

pwd:pwd.c
	$(CC) -o $@ $<

shadow:shadow.c
	$(CC) -o $@ $<

uname:uname.c
	$(CC) -o $@ $<

.PHONY:clean
clean:
	-rm $(TARGETS)
