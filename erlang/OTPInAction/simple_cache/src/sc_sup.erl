-module(sc_sup).

-behaviour(supervisor).

%% API
-export([start_link/0, start_child/2]).
-export([init/1]).

-define(SERVER, ?MODULE).

start_link() ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, []).

start_child(Value, LeaseTime) ->
    supervisor:start_child(?SERVER, [Value, LeaseTime]).

init(_Args) ->
    SupervisorSpecification = #{
        strategy => simple_one_for_one,
        intensity => 0,
        period => 1
    },

    ChildSpecifications = [
        #{
            id => sc_element,
            start => {sc_element, start_link, []},
            restart => temporary,
            shutdown => brutal_kill,
            type => worker, 
            modules => [sc_element]
        }
    ],

    {ok, {SupervisorSpecification, ChildSpecifications}}.
