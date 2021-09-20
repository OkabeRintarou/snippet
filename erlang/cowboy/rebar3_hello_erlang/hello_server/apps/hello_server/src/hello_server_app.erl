%%%-------------------------------------------------------------------
%% @doc hello_server public API
%% @end
%%%-------------------------------------------------------------------

-module(hello_server_app).

-behaviour(application).

-export([start/2, stop/1]).

start(_StartType, _StartArgs) ->
	Dispatch = cowboy_router:compile([
		{<<"localhost">>, [
				{<<"/">>, homepage_handler, []},
				{<<"/hello/[:name]">>, hello_handler, []}
		]}
	]),
	{ok, _} = cowboy:start_clear(
				hello_listener,
				[{port, 8080}],
				#{env => #{dispatch => Dispatch}}
			   ),
    hello_server_sup:start_link().

stop(_State) ->
    ok.

%% internal functions
