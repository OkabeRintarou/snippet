module top;
	logic [1:0] sel;
	logic [7:0] a;

	initial begin
		sel = 2'b01;

		case (sel) 
			2'b00: a = 8'h0a;
			2'b01: a = 8'h0b;
			2'b10: a = 8'h0c;
			2'b11: a = 8'h0d;
			default: a = 8'h0e;
		endcase

		$display("sel = 01, a = %x",a);

		sel = 2'bxx;

		case (sel)
			2'b00: a = 8'h0a;
			2'b01: a = 8'h0b;
			2'b10: a = 8'h0c;
			2'b11: a = 8'h0d;
			default: a = 8'h0e;
		endcase

		$display("sel = xx, a = %x",a);

		sel = 2'bx0;
		case (sel)
			2'b00: a = 8'h0a;
			2'b01: a = 8'h0b;
			2'b10: a = 8'h0c;
			2'b11: a = 8'h0d;
			2'bx0: a = 8'h0e;
			2'bxx: a = 8'h0f;
			default: a = 8'h00;
		endcase

		$display("sel = x0, a = %x",a);
	end
endmodule
