let last lst =
    List.nth lst ((List.length lst) - 1)

let any_zeros lst =
    List.exists (fun a -> a = 0) lst
