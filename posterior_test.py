#!/usr/bin/env python3
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
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

    # Plot accuracy results
    plot_accuracy_results(out_csv, theta_true, est_mean, est_std, n_params)


def plot_accuracy_results(csv_path: str, theta_true, est_mean, est_std, n_params: int):
    """
    Plot true vs estimated parameter values with uncertainty bands.
    Creates one plot per parameter and saves independently.
    
    Args:
        csv_path: Path to the CSV file (used to determine output directory)
        theta_true: Tensor of true parameter values [N, n_params]
        est_mean: List of estimated means [N tensors of shape [n_params]]
        est_std: List of estimated stds [N tensors of shape [n_params]]
        n_params: Number of parameters
    """
    csv_path = Path(csv_path)
    output_dir = csv_path.parent
    
    # Convert to numpy arrays
    true_np = theta_true.detach().cpu().numpy()  # [N, n_params]
    est_mean_np = torch.stack(est_mean).detach().cpu().numpy()  # [N, n_params]
    est_std_np = torch.stack(est_std).detach().cpu().numpy()  # [N, n_params]
    
    param_names = [r"$\kappa$", r"$d_\tau$"] if n_params == 2 else [rf"$\theta_{{{j}}}$" for j in range(n_params)]
    param_file_names = ["kappa", "d_tau"] if n_params == 2 else [f"param_{j}" for j in range(n_params)]
    
    for j in range(n_params):
        true_j = true_np[:, j]
        est_j = est_mean_np[:, j]
        std_j = est_std_np[:, j]
        
        # Sort by true values for better visualization
        sort_idx = np.argsort(true_j)
        true_sorted = true_j[sort_idx]
        est_sorted = est_j[sort_idx]
        std_sorted = std_j[sort_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # Plot true vs estimated
        ax.plot(true_sorted, true_sorted, 'k--', linewidth=1.5, label='Perfect fit', alpha=0.6)
        ax.plot(true_sorted, est_sorted, 'o-', markersize=5, linewidth=1.5, label='Estimated', color='C0')
        
        # Add uncertainty band (±1 std)
        ax.fill_between(
            true_sorted,
            est_sorted - std_sorted,
            est_sorted + std_sorted,
            alpha=0.3,
            color='C0',
            label='±1 std'
        )
        
        ax.set_xlabel(f'True {param_names[j]}', fontsize=12)
        ax.set_ylabel(f'Estimated {param_names[j]}', fontsize=12)
        ax.set_title(f'{param_names[j]}: True vs Estimated', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Equal aspect for comparison
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # Save figure in PDF format (vector format with transparency support)
        output_path = output_dir / f"accuracy_plot_{param_file_names[j]}.pdf"
        plt.savefig(output_path, format='pdf', dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
        plt.close(fig)