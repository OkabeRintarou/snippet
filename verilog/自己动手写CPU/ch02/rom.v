module rom(
	input logic ce,
	input logic[5:0] addr,
	output logic[31:0] inst
);

	logic [31:0] rom[0:63];

	initial $readmemh("./rom.data", rom);

	always @(*) begin
		if (ce == 1'b0) begin
			inst <= 32'h0;
		end else begin
			inst <= rom[addr];
		end
	end
endmodule
