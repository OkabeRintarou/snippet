defmodule NoSlides.MixProject do
  use Mix.Project

  def project do
    [
      app: :no_slides,
      version: "0.1.0",
      elixir: "~> 1.14-dev",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger, :riak_core],
      mod: {NoSlides.Application, []}
    ]
  end

  defp deps do
    [
      {:riak_core, "~> 0.10.4", hex: :riak_core_lite},
    ]
  end
end
