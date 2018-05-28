calc:calc.tab.o lex.yy.o util.o main.o
	gcc -o $@ lex.yy.o calc.tab.o util.o main.o -lm

lex.yy.o:calc.l
	flex calc.l
	gcc -c lex.yy.c

calc.tab.o:calc.y
	bison -d calc.y
	gcc -c calc.tab.c

util.o:util.c
	gcc -c -o util.o util.c

main.o:main.c
	gcc -c -o main.o main.c
.PHONY:clean

clean:
	-rm *.o calc lex.yy.c calc.tab.[ch]

