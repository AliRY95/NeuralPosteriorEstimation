import torch
from simulator_trajectory import population_summary_timeseries


def simulator_trajectory(theta: torch.Tensor, sim_cfg: dict) -> torch.Tensor:
	"""
	Facade: simulate population summary time-series for given theta batch.

	theta: torch.Tensor of shape (N, 2) with [kappa, d_tau].
	sim_cfg: dict configuration forwarded to population_summary_timeseries.

	Returns: torch.Tensor of shape (N, T) with summary time-series per theta.
	"""
	theta = theta.reshape(-1, 2)
	traces = []
	for params in theta:
		kappa = float(params[0])
		d_tau = float(params[1])
		sim_theta = {"kappa": kappa, "d_tau": d_tau}
		series = population_summary_timeseries(sim_theta, sim_cfg)
		traces.append(torch.tensor(series, dtype=torch.float32))
	return torch.stack(traces)
