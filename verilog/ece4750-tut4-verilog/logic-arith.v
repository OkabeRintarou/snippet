module top;

	logic [7:0] A;
	logic [7:0] B;
	logic [7:0] C;
	logic [7:0] D;
	logic [7:0] E;
	logic [7:0] F;

	initial begin
		A = 8'd200;
		B = 8'd400;

		C = A + B;
		D = C + B;
		E = D - B;
		F = E - B;
		$display("200 + 400 + 400 - 400 - 400 = %d",F);
	end
endmodule
