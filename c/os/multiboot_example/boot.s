.set ALIGN, 1<<0
.set MEMINFO, 1<<1
.set FLAGS, ALIGN | MEMINFO
.set MAGIC, 0x1BADB002
.set CHECKSUM, -(MAGIC + FLAGS)


/* multiboot header */
.section .multiboot
.align 4
.long MAGIC
.long FLAGS
.long CHECKSUM


.section .bss
.align 16
stack_bottom:
.skip 16384 # 16KB
stack_top:

.section .text
.globl _start
.type _start, @function
_start:
	/* 
	The bootloader has loaded us into 32-bit protected mode on a x86 
	machine. Interrupts are disabled. Paging is disabled.
	*/
	mov $stack_top, %esp

	/*
	This is a good place to initialize crucial processor state before the high-level
	kernel is entered. Note the processor is not fully initialized yet: Features such
	as floating point instructions and instruction set extensions are not initialized
	yet. The GDT should be loaded here. Paging should be enabled here. C++ features
	such as global constructors and exceptions will require runtime support to work as
	wel
	*/

	/*
	Enter the high-level kernel.
	*/
	call kernel_main

	/*
	If the system has nothing more to do, put the computer into an
	infinite loop. To do that:
	(1) Disable interrupts with cli 
	(2) Wait for the next interrupts to arrive with hlt. Since they are disabled,
	this will lock up the computer.
	(3) Jump to the hlt instruction if it ever wakes up due to a non-maskable interrupt
	occuring or due to system management mode.
	*/
	cli
1:	
	hlt
	jmp 1b

.size _start, . - _start
