MODULE=kernel_api_example

obj-m+=$(MODULE).o 
$(MODULE)-objs := api_demo.o	\
	test_list.o test_spin_lock.o test_dma_fence.o

KERNELDIR=/lib/modules/$(shell uname -r)/build

PWD:=$(shell pwd)
all:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

.PHONY:clean
clean:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean 
