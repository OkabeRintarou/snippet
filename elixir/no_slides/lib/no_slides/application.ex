defmodule NoSlides.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # Starts a worker by calling: NoSlides.Worker.start_link(arg)
      # {NoSlides.Worker, arg}
    ]

    opts = [strategy: :one_for_one, name: NoSlides.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
