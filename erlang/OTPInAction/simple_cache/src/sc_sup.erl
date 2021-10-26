-module(sc_sup).

-behaviour(supervisor).

%% API
-export([start_link/0]).
-export([init/1]).

-define(SERVER, ?MODULE).

start_link() ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, []).

init(_Args) ->
    SupervisorSpecification = #{
        strategy => one_for_one,
        intensity => 4,
        period =>3600
    },

    ElementSup = 
        #{
            id => sc_element_sup, 
            start => {sc_element_sup, start_link, []},
            restart => permanent,
            shutdown => 2000,
            type => supervisor,
            modules => [sc_element]
        },
    EventManager =
        #{
            id => sc_event,
            start => {sc_event, start_link, []},
            restart => permanent,
            shutdown => 2000,
            type => worker,
            modules => [sc_event]
        },

    ChildSpecifications = [ElementSup, EventManager],
    {ok, {SupervisorSpecification, ChildSpecifications}}.
