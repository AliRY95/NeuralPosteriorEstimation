import torch

from simulator_trajectory import population_summary_timeseries


def simulator_trajectory(theta: torch.Tensor, sim_cfg: dict) -> torch.Tensor:
    """
    Facade: simulate population summary time-series for given theta batch and
    convert angle deltas to XY coordinates per timestep.

    theta: torch.Tensor of shape (N, 2) with [kappa, d_tau].
    sim_cfg: dict configuration forwarded to population_summary_timeseries.

    Returns: torch.Tensor of shape (N, T, 2) with XY positions per theta.
    """
    theta = theta.reshape(-1, 2)
    traces = []
    step = float(sim_cfg.get("s", 1.0))
    for params in theta:
        kappa = float(params[0])
        d_tau = float(params[1])
        sim_theta = {"kappa": kappa, "d_tau": d_tau}
        # series: angle deltas per timestep (length T)
        angles = torch.tensor(
            population_summary_timeseries(sim_theta, sim_cfg), dtype=torch.float32
        )
        heading = torch.cumsum(angles, dim=0)
        dx = torch.cos(heading) * step
        dy = torch.sin(heading) * step
        x = torch.cumsum(dx, dim=0)
        y = torch.cumsum(dy, dim=0)
        xy = torch.stack([x, y], dim=-1)  # (T, 2)
        traces.append(xy)
    return torch.stack(traces)
