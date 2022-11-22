`timescale 1 ns/10 ps

module half_adder_tb;
	reg a, b;
	wire sum, carry;

	localparam period = 20;

	half_adder UTT(.a(a), .b(b), .sum(sum), .carry(carry));

	initial // initial block executes only once
	begin
		$dumpfile("wave.vcd");
		$dumpvars(0, half_adder_tb);

		a = 0;
		b = 0;
		#period;  // wait for period

		a = 0;
		b = 1;
		#period;

		a = 1;
		b = 0;
		#period;

		a = 1;
		b = 1;
		#period;
	end
endmodule
