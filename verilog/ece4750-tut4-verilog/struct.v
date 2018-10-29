typedef struct packed {  // Packed format
	logic [3:0] x;				 // 11  8 7  4 3   0
	logic [3:0] y;         // +----+----+----+
	logic [3:0] z;         // | x  | y  | z  |
} point_t;               // +----+----+----+

typedef struct packed {
	logic [7:0] red;
	logic [7:0] green;
	logic [7:0] blue;
}rgb_t;

module top;
	point_t pa;
	point_t pb;
	rgb_t color;

	logic [$bits(point_t) - 1:0] pbits;

	initial begin
		pa.x = 4'h3;
		pa.y = 4'h4;
		pa.z = 4'h5;

		$display("pa.x = %x",pa.x);
		$display("pa.y = %x",pa.y);
		$display("pa.z = %x",pa.z);

		pb = pa;

		$display("pb.x = %x",pb.x);
		$display("pb.y = %x",pb.y);
		$display("pb.z = %x",pb.z);

		pbits = pa;
		$display("pbits = %x",pbits);

		// Assign bit vector to struct
		pbits = {4'd13,4'd9,4'd3};
		pa = pbits;

		$display("pa.x = %x",pa.x);
		$display("pa.y = %x",pa.y);
		$display("pa.z = %x",pa.z);


		color.red = 8'd111;
		color.green = 8'd222;
		color.blue = 8'd255;

		$display("color.red = %x",color.red);
		$display("color.green = %x",color.green);
		$display("color.b;lue = %x",color.blue);
	end
endmodule
