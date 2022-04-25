MODULE=kernel_api_example

obj-m+=$(MODULE).o 
$(MODULE)-objs := api_demo.o	\
	test_list.o test_spin_lock.o test_dma_fence.o test_kmem_cache.o \
	test_percpu.o test_container_of.o

PWD:=$(shell pwd)
KERNEL_DIR ?=/lib/modules/$(shell uname -r)/build
BUILD_DIR ?= $(PWD)/build
BUILD_DIR_MAKEFILE ?= $(PWD)/build/Makefile

all:
	$(MAKE) -C $(KERNEL_DIR) M=$(BUILD_DIR) src=$(PWD) modules

.PHONY:clean
clean:
	$(MAKE) -C $(KERNEL_DIR) M=$(BUILD_DIR) src=$(PWD) clean 
