CC = gcc

TARGETS = file_type file_id access umask chmod link walk chdir dev

all : $(TARGETS)

file_type : file_type.c 
	$(CC) -o $@ $<

file_id : file_id.c 
	$(CC) -o $@ $<

access : access.c 
	$(CC) -o $@ $<

umask : umask.c 
	$(CC) -o $@ $<

chmod : chmod.c 
	$(CC) -o $@ $<

link : link.c
	$(CC) -o $@ $<

walk:walk.c
	$(CC) -o $@ $<

chdir:chdir.c
	$(CC) -o $@ $<

dev : dev.c
	$(CC) -o $@ $<

.PHONY : clean

clean: 
	-rm $(TARGETS)
