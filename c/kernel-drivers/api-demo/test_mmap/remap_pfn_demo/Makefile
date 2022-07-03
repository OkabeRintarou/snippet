MODULE=remap_pfn_alloc_pages

obj-m+=$(MODULE).o 
#$(MODULE)-objs := remap_pfn_alloc_pages.o

PWD:=$(shell pwd)
KERNEL_DIR ?=/lib/modules/$(shell uname -r)/build
BUILD_DIR ?= $(PWD)/build
BUILD_DIR_MAKEFILE ?= $(PWD)/build/Makefile

all:
	$(MAKE) -C $(KERNEL_DIR) M=$(BUILD_DIR) src=$(PWD) modules

.PHONY:clean
clean:
	$(MAKE) -C $(KERNEL_DIR) M=$(BUILD_DIR) src=$(PWD) clean 
