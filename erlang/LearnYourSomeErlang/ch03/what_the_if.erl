-module(what_the_if).
-compile(export_all).

oh_god(N) ->
    if N =:= 2 -> might_succeed;
    true -> always_does
end.

help_me(Animal) ->
    Talk = 
        if Animal == cat -> "meow";
            Animal == beef -> "mooo";
            Animal == dog -> "bark";
            Animal == tree -> "bark";
            true -> "fgdadfgna"
        end,
    {Animal, "says " ++ Talk ++ "!"}.