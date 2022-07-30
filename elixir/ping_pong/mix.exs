defmodule PingPong.MixProject do
  use Mix.Project

  def project do
    [
      app: :ping_pong,
      version: "0.1.0",
      elixir: "~> 1.14-dev",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {PingPong.Application, []}
    ]
  end

  defp deps do
    [
      {:local_cluster, "~> 1.2", only: [:dev, :test]},
    ]
  end
end
