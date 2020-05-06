type student = {
	first_name : string;
	last_name : string;
	gpa : float;
}

let syl = { first_name = "Okabe"; last_name = "Rintarou"; gpa = 4.0 }
let name s = (s.first_name, s.last_name)
let create first_name last_name gpa =
	{ first_name = first_name; last_name = last_name; gpa = gpa }
