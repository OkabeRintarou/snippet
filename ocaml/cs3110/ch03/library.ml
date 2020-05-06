let fifth lst =
let len = List.length lst in
    if len < 5
    then 0
    else List.nth lst 4

let reverse_sort (lst : int list) : int list =
List.rev (List.sort 
    (fun a b ->
        if a = b then 0
        else if a < b then -1
        else 1
    ) lst)

