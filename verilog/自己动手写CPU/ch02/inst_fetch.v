`include "./pc_reg.v"
`include "./rom.v"

module inst_fetch(
	input logic clk,
	input logic rst,
	output logic[31:0] inst_o
);

	logic [5:0] pc;
	logic rom_ce;

	pc_reg pc_reg0(
		.clk(clk),
		.rst(rst),
		.pc(pc),
		.ce(rom_ce)
	);

	rom rom0(
		.ce(rom_ce),
		.addr(pc),
		.inst(inst_o)
	);
endmodule
