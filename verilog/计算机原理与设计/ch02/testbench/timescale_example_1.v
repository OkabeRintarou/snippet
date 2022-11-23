`timescale 1ns/1ns

module tb;
	logic val;

	initial begin
		$dumpfile("timescale_example_1.vcd");
		$dumpvars(0, tb);
		// Initialize the signal to 0 at time 0 units
		val <= 0;
		// Advance by 1 time unit
		#1 $display("T=%0t At time #1", $realtime);
		val <= 1;
		// Advance by 0.49 time unit 
		#0.49 $display("T=%0t At time #0.49", $realtime);
		val <= 0;
		// Advance by 0.5 time unit
		#0.50 $display("T=%0t At time #0.50", $realtime);
		val <= 1;
		// Advance by 0.51 time unit
		#0.51 $display("T=%0t At time #0.51", $realtime);
		val <= 0;

		// Let simulation run for another 5 time units and exit
		#5 $display("T=%0t End of simulation", $realtime);
	end
endmodule

/*
 * Output: 
 * T=1 At time #1								等待 1 个时间单位, 即 1ns
 * T=1 At time #0.49                            0.49ns 取整为 0ns
 * T=2 At time #0.50							0.5ns 取整为 1ns
 * T=3 At time #0.51							0.51ns 取整为 1ns
 * T=8 End of simulation                   		继续等待 5ns, 总计 8ns
 *
 */
