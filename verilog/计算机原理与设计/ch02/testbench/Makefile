all:half_adder_tb

half_adder_tb: half_adder_tb.v half_adder.v
	iverilog -o $@ half_adder_tb.v half_adder.v

run: half_adder_tb
	vvp -n half_adder_tb -lxt2
	gtkwave wave.vcd

.PHONY:clean
clean:
	rm wave.vcd half_adder_tb
