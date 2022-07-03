BINS:=user1 user2 user3 user4

all: $(BINS)

user1:user1.c
	gcc -o $@ $<

user2:user2.c
	gcc -o $@ $<

user3:user3.c
	gcc -o $@ $<

user4:user4.c
	gcc -o $@ $<

.PHONY:clean
clean:
	rm $(BINS)
