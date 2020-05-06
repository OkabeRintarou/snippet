let rec is_decrease lst =
	match lst with
	| [] -> true
	| [_] -> true
	| x::y::xs -> 
		if x < y then false
		else is_decrease (y::xs)

let rec is_uniodal (lst : int list) : bool =
	match lst with 
	| [] -> true
	| [_] -> true
	| x::y::xs -> 
		if x > y then is_decrease (y::xs)
		else is_uniodal (y::xs)

