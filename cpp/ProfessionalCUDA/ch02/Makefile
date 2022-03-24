BINS=sumArraysOnHost checkDimension defineGridBlock sumArraysOnGPU\
	 checkThreadIndex sumMatrix2D
all: $(BINS)

sumArraysOnHost:sumArraysOnHost.cu
	nvcc -o $@ $<

checkDimension:checkDimension.cu
	nvcc -o $@ $<

defineGridBlock:defineGridBlock.cu
	nvcc -o $@ $<

sumArraysOnGPU:sumArraysOnGPU.cu
	nvcc -o $@ $<

checkThreadIndex:checkThreadIndex.cu
	nvcc -o $@ $<

sumMatrix2D:sumMatrixOnGPU-2D-grid-2D-block.cu
	nvcc -o $@ $<

.PHONY:clean
clean:
	rm $(BINS)
