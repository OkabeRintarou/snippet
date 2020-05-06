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
