`timescale 1 ns/10 ps

module half_adder_procedural_tb;
	reg a, b;
	wire sum, carry;

	localparam period = 20;

	half_adder UTT(.a(a), .b(b), .sum(sum), .carry(carry));
	reg clk;

	initial begin
		$dumpfile("half_adder_procedural_tb.vcd");
		$dumpvars(0, half_adder_procedural_tb);
	end

	always begin
		clk = 1'b1;
		#20;

		clk = 1'b0;
		#20;
	end

	always @(posedge clk)
	begin

		a = 0;
		b = 0;
		#period;  // wait for period
		if (sum != 0 || carry != 0)
			$display("test failed for input combination 00");

		a = 0;
		b = 1;
		#period;
		if (sum != 1 || carry != 0)
			$display("test failed for input combination 01");

		a = 1;
		b = 0;
		#period;
		if (sum != 1 || carry != 0)
			$display("test failed for input combination 10");

		a = 1;
		b = 1;
		#period;
		if (sum != 0 || carry != 1)
			$display("test failed for input combination 11");

		$stop;
	end
endmodule
