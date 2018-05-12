CC=gcc

TARGETS=getid fork vfork wait avoid_zombie race exec

all:$(TARGETS)

getid:getid.c
	$(CC) -o $@ $<

fork:fork.c
	$(CC) -o $@ $<

vfork:vfork.c
	$(CC) -o $@ $<

wait:wait.c
	$(CC) -o $@ $<

avoid_zombie:avoid_zombie.c
	$(CC) -o $@ $<

race:race.c
	$(CC) -o $@ $<

exec:exec.c
	$(CC) -o $@ $<

.PHONY:clean
clean:
	-rm $(TARGETS)
