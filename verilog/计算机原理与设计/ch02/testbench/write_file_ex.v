`timescale 1ns/10ps

module write_file_ex;
	reg a, b, sum_expected, carry_expected;
	reg[3:0] read_data[0:5];

	integer write_data;
	integer i;

	initial begin
		$readmemb("./adder_data.txt", read_data);
		write_data = $fopen("./write_file_ex.txt");

		for (i = 0; i < 6; i = i + 1) begin
			{a, b, sum_expected, carry_expected} = read_data[i];
			#20;
			$fdisplay(write_data, "%b_%b_%b_%b", a, b, sum_expected, carry_expected);
		end

		$fclose(write_data);
	end
endmodule
