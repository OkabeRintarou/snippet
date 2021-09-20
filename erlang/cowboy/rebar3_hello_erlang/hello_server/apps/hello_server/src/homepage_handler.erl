-module(homepage_handler).
-behavior(cowboy_handler).
-export([init/2]).

init(Req, State) ->
	Req1 = cowboy_req:reply(
			 200,
			 #{<<"content-type">> => <<"text/plain">>},
			 <<"Homepage">>,
			 Req),
	{ok, Req1, State}.
