TARGETS=wc wc_multi include symbol

all:$(TARGETS)

wc:wc.l
	flex -o wc.c wc.l
	gcc -o wc wc.c -lfl

wc_multi:wc_multi.l
	flex -o wc_multi.c wc_multi.l
	gcc -o $@ wc_multi.c

include:include.l
	flex -o include.c include.l
	gcc -o $@ include.c

symbol:symbol.l
	flex -o symbol.c symbol.l
	gcc -o $@ symbol.c

.PHONY:clean

clean:
	-rm $(TARGETS) wc.c wc_multi.c include.c symbol.c
