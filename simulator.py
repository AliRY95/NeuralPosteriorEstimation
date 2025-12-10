import numpy as np
import torch

from simulator_trajectory import simulate_population


def simulator_trajectory(theta: torch.Tensor, sim_cfg: dict) -> torch.Tensor:
    """
    Simulate raw trajectories for a batch of parameters.
    
    theta: (N, 2) with [kappa, d_tau].
    sim_cfg: dict forwarded to simulate_population.
    
    Returns: torch.Tensor of shape (N, T, 3) with [x(t), y(t), t_norm] per theta.
    """
    theta = theta.reshape(-1, 2)

    T_max = float(sim_cfg.get("T_max", 100.0))
    M = int(np.floor(T_max))
    # normalized time 0..1 (length M)
    t_norm = np.linspace(1.0, float(M), M, dtype=np.float32)
    t_norm = t_norm / float(M)

    traces = []
    step = float(sim_cfg.get("s", 1.0))
    for params in theta:
        kappa = float(params[0])
        d_tau = float(params[1])
        sim_theta = {"kappa": kappa, "d_tau": d_tau}
        # series: angle deltas per timestep (length T)
        traj = simulate_population(sim_theta, sim_cfg)
        tr = traj[0].interpolate_to_integers(T_max)
        # stack into (M, 3): x, y, t_norm
        arr = np.stack(
            [tr.x.astype(np.float32),
             tr.y.astype(np.float32),
             t_norm],
            axis=-1,
        )
        traces.append(torch.from_numpy(arr))
    
    return torch.stack(traces)
