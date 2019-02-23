.data
hello:
	.string "hello, world!\n"

.text
.globl main
main:
	mov $1, %rax
	mov $2, %rdi
	mov $hello, %rsi
	mov $14, %rdx
	syscall
	mov $1, %rax
	xor %rdi, %rdi
	syscall
	ret
