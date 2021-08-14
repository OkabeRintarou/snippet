-module(calc).
-export([rpn/1, rpn_test/0]).

rpn(L) when is_list(L) ->
	[Res] = lists:foldl(fun rpn/2, [], string:tokens(L, " ")),
	Res.

read(N) ->
	case string:to_float(N) of
		{error, no_float} -> list_to_integer(N);
		{F, _} -> F
	end.

rpn("+", [N1, N2 | S]) -> [N2 + N1 | S];
rpn("-", [N1, N2 | S]) -> [N2 - N1 | S];
rpn("*", [N1, N2 | S]) -> [N2 * N1 | S];
rpn("/", [N1, N2 | S]) -> [N2 / N1 | S];
rpn("^", [N1, N2 | S]) -> [math:pow(N1, N2) | S];
rpn("ln", [N | S]) -> [math:log(N) | S];
rpn("log10", [N | S]) -> [math:log10(N) | S];
rpn(X, Stack) -> [read(X) | Stack].

rpn_test() ->
	5 = rpn("2 3 +"),
	87 = rpn("90 3 -"),
	-4 = rpn("10 4 3 + 2 * -"),
	-2.0 = rpn("10 4 3 + 2 * - 2 /"),
	ok = try
			rpn("90 34 12 33 55 66 + * - +")
		catch
			 error:{badmatch,[_|_]} -> ok
		end.

