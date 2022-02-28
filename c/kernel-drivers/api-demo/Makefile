MODULE=kernel_api_example

obj-m+=$(MODULE).o 
$(MODULE)-objs := test_list.o api_demo.o
KERNELDIR=/lib/modules/$(shell uname -r)/build

PWD:=$(shell pwd)
all:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

.PHONY:clean
clean:
	@rm *.o *.ko *.mod *.order *.symvers *.mod.c *.dwo
