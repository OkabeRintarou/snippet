-module(ping).
-behavior(gen_server).

-export([init/1, handle_call/3, handle_info/2]).
-define(TIMEOUT, 5000).

init(_Args) ->
	{ok, undefined, ?TIMEOUT}.

handle_call(start, _From, LoopState) ->
	{reply, started, LoopState, ?TIMEOUT};
handle_call(pause, _From, LoopState) ->
	{reply, paused, LoopState}.

handle_info(timeout, LoopState) ->
	{_Hour, _Min, Sec} = time(),
	io:format("~2.w~n", [Sec]),
	{noreply, LoopState, ?TIMEOUT}.
