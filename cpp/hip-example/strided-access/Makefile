BINS=benchmark-hip
HIPCC=hipcc


all:$(BINS)

benchmark-hip:benchmark-hip.cc
	$(HIPCC) -o $@ $<

.PHONY:clean
clean:
	rm $(BINS)
