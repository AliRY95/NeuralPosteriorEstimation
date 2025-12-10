#!/usr/bin/env python3
import csv
import numpy as np
import torch
from sbi.utils import BoxUniform

from simulator import simulator_trajectory

def eval_accuracy_trajectory(
    N: int,
    sim_cfg: dict,
    prior: BoxUniform,
    posterior,
    out_csv: str = "data/accuracy_results.csv",
) -> None:
    """
    Evaluate the trained posterior on synthetic or CSV-generated data.

    For each test case:
      - Simulate trajectory from sampled true parameters.
      - Condition posterior on trajectory.
      - Draw posterior samples, compute mean and std as estimates.

    Prints MAE, RMSE, and Bias for each parameter.
    Saves CSV with true, estimated mean, and std for each test case.

    CSV columns: true_param_0, ..., true_param_D, est_param_0, ..., est_param_D, std_param_0, ..., std_param_D
    """
    # Sample true parameters and simulate trajectories
    theta_true = prior.sample((N,))  # [N, D]
    x = simulator_trajectory(theta_true, sim_cfg)  # [N, seq_len, features]

    est_mean, est_std, err = [], [], []
    for i in range(N):
        # Add batch dimension for posterior conditioning
        x_o = x[i].unsqueeze(0)  # [1, seq_len, features]
        samples = posterior.sample(torch.Size([100]), x_o)  # [100, 1, D] or [100, D]
        samples = samples.squeeze(1) if samples.ndim == 3 else samples  # [100, D]
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)

        est_mean.append(mean)
        est_std.append(std)
        err.append(mean - theta_true[i])

    err = torch.stack(err, dim=0)  # [N, D]
    mae = err.abs().mean(dim=0)    # [D]
    rmse = torch.sqrt((err**2).mean(dim=0))  # [D]
    bias = err.mean(dim=0)         # [D]

    # Print metrics for each parameter
    n_params = theta_true.shape[1]
    print("\nPosterior accuracy on synthetic trajectories:")
    print(f"Test cases: {N}")
    print(f"{'Param':<8} {'MAE':>12} {'RMSE':>12} {'Bias':>12}")
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    for j in range(n_params):
        print(
            f"{j:<8} "
            f"{mae[j].item():>12.4f} "
            f"{rmse[j].item():>12.4f} "
            f"{bias[j].item():>12.4f}"
        )

    # Write CSV with true, estimated mean, and std values for each test case
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # Header: true params, estimated means, estimated stds
        header = (
            [f"true_param_{j}" for j in range(n_params)] +
            [f"est_param_{j}" for j in range(n_params)] +
            [f"std_param_{j}" for j in range(n_params)]
        )
        writer.writerow(header)
        for t, e, s in zip(theta_true, est_mean, est_std):
            row = (
                [float(t[j]) for j in range(n_params)] +
                [float(e[j]) for j in range(n_params)] +
                [float(s[j]) for j in range(n_params)]
            )
            writer.writerow(row)
    print(f"Saved true/estimated values to {out_csv}")


