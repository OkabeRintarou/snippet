`timescale 1 ns/ 10 ps
`include "./inst_fetch.v"

module inst_fetch_tb;

	/* 第一阶段: 数据类型说明 */
	logic CLOCK_50;
	logic rst;
	logic [31:0] inst;

	initial
	begin
		$dumpfile("wave.vcd");
		$dumpvars(0, inst_fetch_tb);
	end
	/* 第二阶段: 激励向量定义 */
	initial begin
		CLOCK_50 = 1'b0;
		forever #10 CLOCK_50 = ~CLOCK_50;
	end


	initial begin
		rst = 1'b1;
		#195 rst = 1'b0;
		#1000 $finish;
	end

	/* 第三阶段: 待测试模块例化 */
	inst_fetch inst_fetch0(
		.clk(CLOCK_50),
		.rst(rst),
		.inst_o(inst));

	initial begin
		$monitor("%h", inst_fetch_tb.inst_fetch0.inst_o);
	end
endmodule
