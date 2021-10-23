-module(tr_sup).

-behaviour(supervisor).

%% API
-export([start_link/0]).
-export([init/1]).

-define(SERVER, ?MODULE).

start_link() ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, []).

init(_Args) ->
    RestartStrategy = #{
        strategy => one_for_one, 
        intensity => 0,
        period => 1},

    ChildSpecifications = [
        #{
            id => tr_server,
            start => {tr_server, start_link, []},
            restart => permanent,
            shutdown => 2000,
            type => worker,
            modules => [tr_server]
        }
    ],

    {ok, {RestartStrategy, ChildSpecifications}}.