`timescale 1ns/10ps

module read_file_ex;
	reg a, b;
	reg [1:0] sum_carry_expected;

	reg[3:0] read_data[0:5];
	integer i;

	initial begin
		$dumpfile("read_file_ex.vcd");
		$dumpvars(0, read_file_ex);
		/// readmemb = read the binary values from the file
		$readmemb("adder_data.txt", read_data);
		/// total number of lines in adder_data.txt = 6
		for (i = 0; i < 6; i = i + 1) begin
			// 0_1_1_0 and 0110 are read in the same way, i.e.
			// a = 0, b = 1, sum_expected = 1,  carry_expected = 0 for above line
			{a, b, sum_carry_expected} = read_data[i];
			#20;
		end
	end
endmodule
