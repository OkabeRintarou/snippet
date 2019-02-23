TS=test01 test02 test03 test04 test05 dummy

all:$(TS)

test01:test01.c
	gcc -o $@ $<
test02:test02.c
	gcc -o $@ $<
test03:test03.c
	gcc -o $@ $<
test04:test04.c
	gcc -o $@ $<
test05:test05.c
	gcc -o $@ $<
dummy:dummy.s
	gcc -o $@ $<
.PHONY:clean
clean:
	rm -rf $(TS)
