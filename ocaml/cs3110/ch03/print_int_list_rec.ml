let rec print_int_list = function
	| [] -> ()
	| h::t -> 
		print_int h;
		print_endline "";
		print_int_list t
	
