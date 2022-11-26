`include "add4.v"
`timescale 1ns/10ps

module add4_tb;
	logic [3:0] a, b;
	logic ci;
	logic [3:0] s, s_e;
	logic co, co_e;
	logic [13:0] read_data[0:511];
	
	integer i;

	add4 adder(a, b, ci, s, co);

	initial begin
		$dumpfile("add4_tb.vcd");
		$dumpvars;

		a = 4'b0;
		b = 4'b0;
		ci = 1'b0;
		#5;

		$readmemb("tests/add4_tb.txt", read_data);

		for (i = 0; i < 512; i = i + 1) begin
			{a, b, ci, s_e, co_e} = read_data[i];
			#10;
			if (s_e != s || co_e != co)
				$error("%h+%h+%b = %h(%b), actual: %h%b", a, b, ci, s_e, co_e, s, co);
		end
	end
endmodule
