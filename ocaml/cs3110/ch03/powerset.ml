let rec powerset_impl lst result =
	match lst with
	| [] -> result
	| x::xs -> 
		let new_result = List.map (fun e -> x::e) result in
		powerset_impl xs (result@new_result)

let rec powerset lst = 
	powerset_impl lst [[]]
