-module(frequency).
-behavior(gen_server).

-export([start_link/0, allocate/0, deallocate/1, stop/0]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

start_link() ->
	gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

init(_Args) ->
	Frequencies = {get_frequencies(), []},
	{ok, Frequencies}.

get_frequencies() -> [10, 11, 12, 13, 14, 15].

allocate() ->
	gen_server:call(?MODULE, {allocate, self()}).

handle_call({allocate, Pid}, _From, Frequencies) ->
	{NewFrequencies, Reply} = allocate(Frequencies, Pid),
	{reply, Reply, NewFrequencies}.

allocate({[], Allocated}, _Pid) ->
	{{[], Allocated}, {error, no_frequency}};
allocate({[Freq|Free], Allocated}, Pid) ->
	{{Free, [{Freq, Pid} | Allocated]}, {ok, Freq}}.

deallocate(Freq) ->
	gen_server:cast(?MODULE, {deallocate, Freq}).

handle_cast(stop, LoopState) ->
	{stop, normal, LoopState};
handle_cast({deallocate, Freq}, Frequencies) ->
	NewFrequencies = deallocate(Frequencies, Freq),
	{noreply, NewFrequencies}.

deallocate({Free, Allocated}, Freq) ->
	NewAllocated = lists:keydelete(Freq, 1, Allocated),
	{[Freq | Free], NewAllocated}.

handle_info({'EXIT', _Pid, normal}, LoopState) ->
	{noreply, LoopState};
handle_info({'EXIT', Pid, Reason}, LoopState) ->
	io:format("Process: ~p exited with reason: ~p~n", [Pid, Reason]),
	{noreply, LoopState};
handle_info(_Msg, LoopState) ->
	{noreply, LoopState}.

stop() ->
	gen_server:cast(frequency, stop).

terminate(_Reason, _LoopState) ->
	io:format("terminate called~n"),
	ok.
