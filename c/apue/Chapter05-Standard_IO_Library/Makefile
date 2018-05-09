CC=gcc

TARGETS=buffering tmpfile mkstemp memory_stream
all:$(TARGETS)

buffering:buffering.c
	$(CC) -o $@ $<

tmpfile:tmpfile.c
	$(CC) -o $@ $<

mkstemp:mkstemp.c
	$(CC) -o $@ $<

memory_stream:memory_stream.c
	$(CC) -o $@ $<

.PHONY:clean

clean:
	-rm $(TARGETS)
