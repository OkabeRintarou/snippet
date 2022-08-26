BINS=vec_add
HIPCC=hipcc


all:$(BINS)

vec_add:vec_add.cc
	$(HIPCC) -o $@ $<

.PHONY:clean
clean:
	rm $(BINS)
