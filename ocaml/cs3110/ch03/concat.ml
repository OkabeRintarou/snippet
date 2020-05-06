let rec concat0 (result : string) (strs : string list) : string =
match strs with
| [] -> result
| h::t -> concat0 (result^h) t

let concat (strs: string list) : string =
    concat0 "" strs
