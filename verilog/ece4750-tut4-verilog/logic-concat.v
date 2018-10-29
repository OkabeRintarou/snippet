module top;
	logic [3:0] A;
	logic [3:0] B;
	logic [3:0] C;
	logic [7:0] D;
	logic [11:0] E;

	initial begin
		A = 4'ha;
		B = 4'hb;
		C = 4'hc;
		D = 8'hde;

		E = {A,B,C};	$display("{%x,%x,%x} = %x",A,B,C,E);
		E = {A,D};		$display("{%x,%x} = %x",A,D,E);

		A = 4'ha;
		B = 4'hb;
		E = {3{A}};		$display("{3{%x}} = %x",A,E);
		E = {A,{2{B}}};	$display("{%x,{2{%x}}} = %x",A,B,E);
	end
endmodule
