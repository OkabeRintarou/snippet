all:c06_mbr.bin

%.bin:%.s
	nasm $^ -o $@

image:c06_mbr.bin
	qemu-img create -f raw kernel.img 1m
	dd if=c06_mbr.bin of=kernel.img bs=512 count=1 conv=notrunc

qemu:kernel.img
	qemu-system-i386 -hda kernel.img

.PHONY:
clean:
	@rm *.bin *.img
