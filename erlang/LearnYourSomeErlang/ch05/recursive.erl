-module(recursive).
-compile(export_all).

fac(N) when N =:= 0 -> 1;
fac(N) when N > 0 -> N * fac(N - 1).

len([]) -> 0;
len([_ | T]) -> 1 + len(T).

duplicate(0, _) -> [];
duplicate(N, Term) when N > 0 ->
	[Term | duplicate(N - 1, Term)].

tail_reverse(L) -> tail_reverse(L, []).

tail_reverse([], Acc) -> Acc;
tail_reverse([H | T], Acc) -> tail_reverse(T, [H | Acc]).
