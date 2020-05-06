all:kernel 


kernel:boot.o kernel.o
	ld -m elf_i386 -T linker.ld -o $@  boot.o kernel.o -nostdlib

boot.o:boot.s
	as --32 -o $@ $<

kernel.o:kernel.c
	gcc -c -m32 -o $@ -std=gnu99 -ffreestanding -fno-builtin -O2 -Wall -Wextra $<
	
qemu:kernel
	qemu-system-i386 -kernel kernel

iso:myos.iso
	qemu-system-i386 -cdrom $<

myos.iso:
	rm -rf isodir/
	mkdir -p isodir/boot/grub
	cp kernel isodir/boot/kernel
	cp grub.cfg isodir/boot/grub/grub.cfg
	grub-mkrescue -o myos.iso isodir

.PHONY:clean
clean:
	@rm *.o kernel
	@rm -rf isodir myos.iso
