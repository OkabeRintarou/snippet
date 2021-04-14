obj-m:=demo_chr_dev.o
KERNELDIR=/lib/modules/$(shell uname -r)/build

PWD:=$(shell pwd)
all:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules
	gcc -o main main.c

.PHONY:clean
clean:
	rm *.o *.ko *.mod *.order *.symvers main *.mod.c
