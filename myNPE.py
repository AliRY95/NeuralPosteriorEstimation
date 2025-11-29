#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from sbi.inference import NPE
from sbi.neural_nets import embedding_nets, posterior_nn
from sbi.utils import BoxUniform

from simulator import simulator_trajectory
from posterior_test import eval_accuracy_trajectory


def save_posterior(path: str | Path, prior: BoxUniform, posterior) -> None:
    """Save posterior neural net state and minimal metadata to disk.

    Stores: state_dict, theta_min, theta_max, input_length, model.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    net = posterior._neural_net  # underlying torch Module
    # Infer input length from a sample attribute or prior sims (not stored here);
    # we rely on caller to pass correct value via posterior.x_shape if available.
    input_len = getattr(posterior, "x_shape", None)
    checkpoint = {
        "state_dict": net.state_dict(),
        "theta_min": prior.support.lower_bound.detach().cpu().tolist(),
        "theta_max": prior.support.upper_bound.detach().cpu().tolist(),
        "input_length": input_len,
        "model": "maf",
    }
    torch.save(checkpoint, path)
    # Write sidecar JSON (human-readable)
    meta_path = path.with_suffix(".json")
    with meta_path.open("w") as f:
        json.dump({k: v for k, v in checkpoint.items() if k != "state_dict"}, f, indent=2)
    print(f"[SAVE] Posterior saved to {path} and metadata to {meta_path}")


def load_posterior(path: str | Path, prior: BoxUniform):
    """Load posterior from checkpoint file, reconstructing neural net and wrapper."""
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu")
    density_estimator = posterior_nn(
        model="maf",
        z_score_x="none",
        z_score_y="none",
    )
    density_estimator.load_state_dict(checkpoint["state_dict"])
    inference = NPE(prior=prior, density_estimator=density_estimator)
    posterior = inference.build_posterior(density_estimator)
    print(f"[LOAD] Posterior loaded from {path}")
    return posterior


def main():
    parser = argparse.ArgumentParser(description="Train or load posterior and evaluate.")
    parser.add_argument("--save", type=str, default=None, help="Path to save trained posterior checkpoint (.pt)")
    parser.add_argument("--load", type=str, default=None, help="Path to existing posterior checkpoint to load instead of training")
    parser.add_argument("--n-sim", type=int, default=10000, help="Number of simulations for training if not loading")
    parser.add_argument("--eval-n", type=int, default=200, help="Number of synthetic test draws for accuracy evaluation")
    args = parser.parse_args()
    # Set seed and rng for reproducibility
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Prior over (kappa, d_tau)
    theta_min = torch.tensor([1.0, 0.01])
    theta_max = torch.tensor([8.0, 0.15])
    prior = BoxUniform(theta_min, theta_max)

    # Generate simulations
    sim_cfg = dict(
        N_pop=1,
        T_max=100.0,
        lambdaa=0.5,
        s=1.0,
        theta_init=None,
        seed=seed,
    )
    if args.load:
        # Load posterior directly; still need some sims if later evaluating transformer; skip training for CNN posterior.
        posterior_cnn = load_posterior(args.load, prior)
    else:
        num_simulations = args.n_sim
        thetas = prior.sample((num_simulations,))
        xs = simulator_trajectory(thetas, sim_cfg=sim_cfg)
        density_estimator_cnn = posterior_nn(
            model="maf",
            z_score_x="none",
            z_score_y="none",
        )
        inference_cnn = NPE(prior=prior, density_estimator=density_estimator_cnn)
        posterior_cnn = inference_cnn.append_simulations(thetas, xs).train()
        if args.save:
            save_posterior(args.save, prior, posterior_cnn)

    # CNN posterior evaluation
    eval_accuracy_trajectory(args.eval_n, sim_cfg, prior, posterior_cnn)
    

    # -----------------------------
    # Using TransformerEmbedding
    # -----------------------------
    trans_cfg = dict(
        vit=False,             # Use standard transformer, not ViT-style
        feature_space_dim=192, # Internal embedding dimension (num_heads * head_dim)
        sequence_length=100,   # Number of time points
        output_dim=16,         # Output feature dimension for NPE
        num_layers=3,          # Transformer depth
        num_heads=12,          # Number of attention heads
        head_dim=16,           # Size per attention head
        d_model=192,           # same as feature_space_dim
    )

    base_trans = embedding_nets.TransformerEmbedding(trans_cfg)

    class ProjectedTransformer(nn.Module):
        def __init__(self, transformer):
            super().__init__()
            self.proj, self.transformer = nn.Linear(1, 192), transformer

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 2:
                x = x.unsqueeze(-1)
            x = self.proj(x)
            return self.transformer(x)

    embedding_trans = ProjectedTransformer(base_trans)

    density_estimator_trans = posterior_nn(
        model="maf",
        embedding_net=embedding_trans,
        z_score_x="none",
        z_score_y="none",
    )
    # Train transformer posterior only if we performed training (no load option for it here)
    if not args.load:
        inference_trans = NPE(prior=prior, density_estimator=density_estimator_trans)
        posterior_trans = inference_trans.append_simulations(thetas, xs).train()
        eval_accuracy_trajectory(args.eval_n, sim_cfg, prior, posterior_trans)


if __name__ == "__main__":
    main()
