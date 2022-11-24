`include "add1.v"
`timescale 1ns/1ps

module add1_tb;
	logic a, b, ci;
	logic s_expected, co_expected;
	logic s_real, co_real;
	logic[4:0] read_data[0:7];

	add1 adder(a, b, ci, s_real, co_real);
	
	integer i;

	initial begin
		$dumpfile("add1_tb.vcd");
		$dumpvars;

		$readmemb("tests/add1_tb.txt", read_data);

		for (i = 0; i < 8; i = i + 1) begin
			{a, b, ci, s_expected, co_expected} = read_data[i];
			#10;
			if (s_expected != s_real || co_expected != co_real)
				$display("%b+%b+%b [actual: %b%b] [expected: %b%b]", a, b, ci, co_real, s_real, co_expected, s_expected);
		end
	end
endmodule
