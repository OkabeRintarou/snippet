obj-m:=hello.o
KERNELDIR=/lib/modules/$(shell uname -r)/build

PWD:=$(shell pwd)
all:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

.PHONY:clean
clean:
	@rm *.o *.ko *.mod *.order *.symvers *.mod.c *.dwo
