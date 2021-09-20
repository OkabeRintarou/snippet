-module(hello_handler).
-behavior(cowboy_handler).
-export([init/2]).

init(Req, State) ->
	Req1 = cowboy_req:reply(
			 200,
			 #{<<"content-type">> => <<"text/plain">>},
			 erlang:iolist_to_binary([<<"hello, ">>, cowboy_req:binding(name, Req, "World")]),
			 Req),
	{ok, Req1, State}.
