BINS=modeset_single_buffer

all:$(BINS)

modeset_single_buffer:modeset_single_buffer.cpp
	g++ -g -std=c++17 -I/usr/include/libdrm -o $@ $< -ldrm

.PHONY:clean
clean:
	@rm $(BINS)
