all: hello

hello:hello.o
	ld -o hello -e mystart hello.o


hello.o:hello.S
	nasm -g -f elf64 hello.S

.PHONY:clean
clean:
	@rm hello hello.o
