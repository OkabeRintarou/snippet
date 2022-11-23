BINS=half_adder_tb \
	 timescale_example_1 \
	 timescale_example_2 \
	 timescale_example_3

all:$(BINS)

half_adder_tb: half_adder_tb.v half_adder.v
	iverilog -o $@ half_adder_tb.v half_adder.v
timescale_example_1:timescale_example_1.v
	iverilog -o $@ $<
timescale_example_2:timescale_example_2.v
	iverilog -o $@ $<
timescale_example_3:timescale_example_3.v
	iverilog -o $@ $<

run: half_adder_tb
	vvp -n half_adder_tb -lxt2
	gtkwave wave.vcd

.PHONY:clean
clean:
	rm $(BINS) *.vcd
