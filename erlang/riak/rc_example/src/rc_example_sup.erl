%%%-------------------------------------------------------------------
%% @doc rc_example top level supervisor.
%% @end
%%%-------------------------------------------------------------------

-module(rc_example_sup).

-behaviour(supervisor).

-export([start_link/0]).

-export([init/1]).

-define(SERVER, ?MODULE).

start_link() ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, []).

init([]) ->
	VMaster = {rc_example_vnode_master,
			   {riak_core_vnode_master, start_link, [rc_example_vnode]},
			   permanent, 5000, worker, [riak_core_vnode_master]},
    {ok, {{one_for_one, 5, 10}, [VMaster]}}.

%% internal functions
