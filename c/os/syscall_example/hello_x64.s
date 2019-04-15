.text

Hello:
	.ascii "Hello, World!\n"

.globl main
main:
	mov $0x01, %rax
	mov $0x01, %rdi
	mov $Hello, %rsi
	mov $0x0e, %rdx
	syscall
