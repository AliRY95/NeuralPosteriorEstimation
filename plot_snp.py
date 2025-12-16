#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt

def generate_perturbed_thetas(
    theta_star,
    prior,
    radii,          
    direction=None, 
):
    device = theta_star.device
    theta_min, theta_max = prior.low.to(device), prior.high.to(device)
    width = theta_max - theta_min
    d = theta_star.numel()

    # fixed direction
    if direction is None:
        v = torch.randn(d, device=device)
        v = v / (v.norm() + 1e-12)
    else:
        v = direction / (direction.norm() + 1e-12)

    thetas = []
    for r in radii:
        theta = theta_star + r * width * v
        theta = torch.max(torch.min(theta, theta_max), theta_min)
        thetas.append(theta)

    return thetas, v

def local_inverse_experiment(
    posterior,
    simulator,
    prior,
    sim_cfg,
    theta_star,
    radii,
    flatten_x,
):
    thetas, direction = generate_perturbed_thetas(
        theta_star, prior, radii
    )

    results = []

    for r, theta in zip(radii, thetas):
        x = simulator(theta.unsqueeze(0), sim_cfg)
        x_infer = x.reshape(1, -1) if flatten_x else x

        with torch.no_grad():
            samples = posterior.sample((2000,), x=x_infer)

        theta_hat = samples.mean(0)

        results.append({
            "r": r,
            "theta_true": theta,
            "theta_hat": theta_hat,
            "traj_true": simulator(theta.unsqueeze(0), sim_cfg)[0],
            "traj_hat": simulator(theta_hat.unsqueeze(0), sim_cfg)[0],
        })

    return results

def plot_local_inverse_results(results, field_type, out_path=None):
    n = len(results)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 4.2 * nrows),
        squeeze=False
    )

    xs, ys = [], []
    for res in results:
        xs.extend(res["traj_true"][:, 0])
        ys.extend(res["traj_true"][:, 1])
        xs.extend(res["traj_hat"][:, 0])
        ys.extend(res["traj_hat"][:, 1])

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    for i, res in enumerate(results):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        t_true = res["traj_true"]
        t_hat  = res["traj_hat"]

        ax.plot(
            t_true[:, 0], t_true[:, 1],
            color="black", lw=2.2, label="true"
        )
        ax.plot(
            t_hat[:, 0], t_hat[:, 1],
            color="red", lw=2.0, ls="--", label="posterior mean"
        )

        theta_true = res["theta_true"]
        theta_hat  = res["theta_hat"]

        if len(theta_true) == 2:
            names = ["κ", "dτ"]
        elif len(theta_true) == 3:
            names = ["κ", "dτ", "L"]
        else:
            names = [f"θ{i}" for i in range(len(theta_true))]


        errs = [
            abs(theta_true[i] - theta_hat[i]).item()
            for i in range(len(theta_true))
        ]
        sci = lambda x: f"{x:.2e}"

        err_str = ", ".join(
            rf"$\Delta {names[i]} = {sci(errs[i])}$"
            for i in range(len(errs))
        )

        ax.set_title(
            rf"$r = {res['r']:.2f}$" + "\n" + err_str,
            fontsize=10
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.2)

        if i == 0:
            ax.legend(frameon=False, fontsize=9)

    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"{field_type}",
        fontsize=14,
        y=0.97
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if out_path:
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()

def local_inverse_trajectory_test(
    posterior,
    simulator,
    theta_star,
    prior,
    sim_cfg,
    x_obs,
    flatten_x,
    n_tests=4,
    n_post_samples=3000,
):
    device = theta_star.device
    theta_min = prior.low.to(device)
    theta_max = prior.high.to(device)
    width = theta_max - theta_min
    d = theta_star.numel()

    v = torch.randn(d, device=device)
    v = v / (v.norm() + 1e-12)

    radii = torch.linspace(0.0, 0.24, n_tests, device=device)

    results = []

    for r in radii:
        theta_true = theta_star + r * width * v
        theta_true = torch.max(torch.min(theta_true, theta_max), theta_min)

        # simulate trajectory
        traj_true = simulator(theta_true.unsqueeze(0), sim_cfg)
        x_infer = traj_true.reshape(1, -1) if flatten_x else traj_true

        # posterior inference
        with torch.no_grad():
            samples = posterior.sample(
                (n_post_samples,),
                x=x_infer
            )

        theta_hat = samples.mean(0)

        traj_hat = simulator(theta_hat.unsqueeze(0), sim_cfg)

        results.append({
            "r": float(r.item()),
            "theta_true": theta_true.detach(),
            "theta_hat": theta_hat.detach(),
            "traj_true": traj_true[0].detach().cpu().numpy(),
            "traj_hat": traj_hat[0].detach().cpu().numpy(),
        })

    return results
