.text

Hello:
	.ascii "Hello, World!\n"

.globl main
main:
	mov $0x04, %eax
	mov $0x01, %ebx
	mov $Hello, %ecx
	mov $0x0e, %edx
	int $0x80
