let pattern0 (strs : string list) : bool =
match strs with
| [] -> false
| h::_ -> h = "bigred"

let pattern1 lst =
match lst with
| _::_::[] | _::_::_::_::[] -> true
| _ -> false

let pattern2 lhs rhs = 
match lhs, rhs with
| [], []
| [], _ 
| _, []
| _::[], _
| _, _::[] -> false
| hx0::hx1::_, hy0::hy1::_ -> hx0 = hy0 && hx1 = hy1
