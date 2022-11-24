BINS=add1_tb

all:$(BINS)

add1_tb:add1_tb.v 
	iverilog -g2012 -o $@ $<

.PHONY:clean
clean:
	rm *.vcd $(BINS)
