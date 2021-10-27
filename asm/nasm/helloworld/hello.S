section .text

global mystart

mystart:
	; prepare the arguments for the system call to write
	; call write(1, mymsg, mylen);
	;           rdx  rsi    rdx
	mov rax, 1
	mov rdx, 1
	mov rsi, mymsg
	mov rdx, mylen
	syscall


	; exit the program
	mov rax, 60
	mov rdx, 0
	syscall

section .data
	mymsg db "hello, world", 0xa
	mylen equ $-mymsg
	
