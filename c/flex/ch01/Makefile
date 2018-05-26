CC=gcc
LINKS=-lfl

TARGETS=example wc calc

all:$(TARGETS)

example:example.l
	flex -o example.c $<
	$(CC) -o $@ example.c $(LINKS)

wc:wc.l
	flex -o wc.c $<
	$(CC) -o $@ wc.c $(LINKS)

calc:calc.l calc.y
	flex calc.l
	bison -d calc.y
	$(CC) -o $@ calc.tab.c lex.yy.c $(LINKS)

.PHONY:clean

clean:
	-rm $(TARGETS) example.c wc.c *.tab.h *.tab.c *.yy.c
