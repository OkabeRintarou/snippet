let rec take_impl cur cur_lst n input_lst =
	if cur = n
	then cur_lst
	else
		match input_lst with
		| [] -> cur_lst
		| x::xs -> take_impl (cur + 1) (cur_lst @ [x]) n xs
		

let take n lst =
    take_impl 0 [] n lst

let rec drop_impl cur n input_lst =
	if cur = n then input_lst
	else
		match input_lst with
		| [] -> input_lst
		| _::xs -> drop_impl (cur + 1) n xs
let drop n lst =
	drop_impl 0 n lst

