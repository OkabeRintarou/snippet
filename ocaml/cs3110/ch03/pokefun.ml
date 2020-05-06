type poketype = Normal | Fire | Water

type pokemon = {
	name : string;
	hp : int;
	ptype : poketype;
}

let r1 = { 
	name = "charizard";
	hp = 78; 
	ptype = Fire;
}

let r2 = {
	name = "squirtle";
	hp = 44;
	ptype = Water;
} 

let rec max_hp lst =
	match lst with
	| [] -> None
	| h::t ->
		match max_hp t with
		| None -> Some h
		| Some m -> Some (if h.hp < m.hp then m else h)
