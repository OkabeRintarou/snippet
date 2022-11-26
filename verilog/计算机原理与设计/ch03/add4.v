`include "add1.v"
module add4(input logic[3:0] a, b, 
			input logic ci,
			output logic [3:0] s,
			output logic co);

	logic [2:0] cout;

	add1 adder0(a[0], b[0], ci, s[0], cout[0]);
	add1 adder1(a[1], b[1], cout[0], s[1], cout[1]);
	add1 adder2(a[2], b[2], cout[1], s[2], cout[2]);
	add1 adder3(a[3], b[3], cout[2], s[3], co);
endmodule
