let rec product (lst : int list) : int =
match lst with
| [] -> 1
| h::t -> h * product t
