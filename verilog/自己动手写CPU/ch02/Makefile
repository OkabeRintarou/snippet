inst_fetch_tb:inst_fetch_tb.v
	iverilog -g2012 -o inst_fetch_tb inst_fetch_tb.v

run: inst_fetch_tb
	vvp -n inst_fetch_tb -lxt2
	gtkwave wave.vcd

.PHONY:clean

clean:
	@rm -rf inst_fetch_tb *.vcd

