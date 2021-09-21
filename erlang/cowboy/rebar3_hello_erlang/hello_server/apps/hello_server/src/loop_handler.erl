-module(loop_handler).
-behavior(cowboy_handler).
-export([init/2]).

init(Req, State) ->
	{cowboy_loop, Req, State}.

info({reply, Body}, Req, State) ->
	pass;
info(_Msg, Req, State) ->
	pass.
