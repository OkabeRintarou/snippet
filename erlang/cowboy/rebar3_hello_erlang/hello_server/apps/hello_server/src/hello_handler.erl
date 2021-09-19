-module(hello_handler).
-behavior(cowboy_handler).
-export([init/2]).

init(Req, State) ->
	Req1 = cowboy_req:reply(
			 200,
			 #{<<"content-type">> => <<"text/plain">>},
			 <<"hello,world">>,
			 Req),
	{ok, Req1, State}.
