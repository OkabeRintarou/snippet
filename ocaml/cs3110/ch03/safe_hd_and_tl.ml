let safe_hd lst =
	match lst with
	| [] -> None
	| h::_ -> Some h
let safe_tl lst =
	match lst with
	| [] -> None
	| _::t -> Some t
