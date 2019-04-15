all:hello_x86 hello_x64

hello_x86:hello_x86.s
	gcc -m32 -o $@ $<

hello_x64:hello_x64.s
	gcc -o $@ $<

.PHONY:clean
clean:
	@rm -rf hello_x86 hello_x64
