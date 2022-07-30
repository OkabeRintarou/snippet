defmodule PingPong.Consumer do
  use GenServer

  alias PingPong.Producer

  @initial %{counts: %{}}

  @impl true
  def init(_arg) do
    Process.send_after(self(), :catch_up, 400)
    {:ok, @initial}
  end

  def start_link(args) do
    GenServer.start_link(__MODULE__, args, name: __MODULE__)
  end

  def total_pings(server) do
    GenServer.call(server, :total_pings)
  end

  def count_for_node(server \\ __MODULE__, node) do
    counts = GenServer.call(server, :get_pings)
    counts[node]
  end

  @impl true
  def handle_cast({:ping, index, node}, data) do
    {:noreply, put_in(data, [:counts, node], index)}
  end

  @impl true
  def handle_call(:get_pings, _from, data) do
    {:reply, data.counts, data}
  end

  @impl true
  def handle_call(:total_pings, _from, data) do
    ping_count =
      data.counts
      |> Enum.map(fn {_, count} -> count end)
      |> Enum.sum()
    {:reply, ping_count, data}
  end

  @impl true
  def handle_call(:flush, _, _) do
    {:reply, :ok, @initial}
  end

  @impl true
  def handle_call(:crash, _from, _data) do
    _count = 42 / 0
    {:reply, :ok, @initial}
  end

  @impl true
  def handle_info(:catch_up, data) do
    GenServer.abcast(Producer, {:catch_up, self()})
    {:noreply, data}
  end
end
