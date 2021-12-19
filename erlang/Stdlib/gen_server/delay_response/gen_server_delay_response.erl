-module(gen_server_delay_response).

-behaviour(gen_server).

-export([delay/2]).
-export([start_link/0, stop/0]).
-export([init/1, handle_call/3, handle_cast/2, terminate/2, code_change/3]).

-define(SERVER, ?MODULE).

%% APIs
delay(Milliseconds, EchoValue) ->
    gen_server:call(?SERVER, {delay_request, Milliseconds, EchoValue}).

start_link() ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, [], []).

stop() ->
    gen_server:call(?SERVER, stop).

%% gen_server callbacks
init([]) ->
    {ok, []}.

handle_call(stop, _From, State) ->
    {stop, normal, stopped, State};
handle_call({delay_request, Milliseconds, EchoValue}, From, State) ->
    spawn_link(fun() -> handle_delay(From, Milliseconds, EchoValue) end),
    {noreply, State}.

handle_cast(_Msg, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

handle_delay(From, Milliseconds, EchoValue) ->
    timer:sleep(Milliseconds),
    gen_server:reply(From, EchoValue).
