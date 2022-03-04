BINS:=dma_fence_client

all:$(BINS)

dma_fence_client:dma_fence_client.c
	gcc -o $@ $< -lpthread

.PHONY:clean
clean:
	@rm $(BINS)
