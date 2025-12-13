#!/usr/bin/env python3

import argparse

import numpy as np
import torch
from sbi.inference import NPE
from sbi.neural_nets import embedding_nets, posterior_nn
from sbi.utils import BoxUniform
from torch import nn

from restartNetwork import (
    load_posterior,
    load_posterior_transformer,
    save_posterior,
    save_posterior_transformer,
)
from posterior_test import eval_accuracy_trajectory
from simulator import simulator_trajectory


def main():
    parser = argparse.ArgumentParser(description="Train or load posterior and evaluate (SNPE).")
    parser.add_argument(
        "-n",
        "--n-sim",
        type=int,
        default=10000,
        help="Total number of simulations (split across SNPE rounds)",
    )
    parser.add_argument(
        "-l",
        "--load",
        type=str,
        default=None,
        help="Path to existing posterior checkpoint to load instead of training",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        default=None,
        help="Path (file or dir) to save trained posterior checkpoint",
    )
    parser.add_argument(
        "-t",
        "--transformer",
        action="store_true",
        help="Use transformer embedding; default is basic MAF on flattened XY",
    )
    parser.add_argument(
        "-r",
        "--rounds",
        type=int,
        default=4,
        help="Number of SNPE rounds",
    )
    args = parser.parse_args()
    
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Prior
    theta_min = torch.tensor([1.0, 0.01])
    theta_max = torch.tensor([8.0, 0.15])
    prior = BoxUniform(theta_min, theta_max)

    sim_cfg = dict(
        N_pop=1,
        T_max=100.0,
        lambdaa=0.5,
        s=1.0,
        theta_init=None,
        seed=seed,
    )

    # ----------------------------------------------------------------
    # "observed data" x0   # should add option to load from file later :D
    # ----------------------------------------------------------------
    true_theta = torch.tensor([3.0, 0.12])
    x_obs = simulator_trajectory(true_theta.unsqueeze(0), sim_cfg)

    # print("x_obs shape:", x_obs.shape)
    # print("true theta:", true_theta)

    # ----------------------------------------------------------------
    # SNPE setup
    # ----------------------------------------------------------------
    num_rounds = args.rounds
    sims_per_round = args.n_sim // num_rounds

    print(f"\nSNPE SETTINGS:")
    print(f"  rounds            = {num_rounds}")
    print(f"  sims_per_round    = {sims_per_round}")
    print(f"  total simulations = {sims_per_round * num_rounds}")
    print("-" * 60)

    # ----------------------------------------------------------------
    #  choice of embedding
    # ----------------------------------------------------------------
    if args.transformer:
        if args.load:
            posterior = load_posterior_transformer(args.load, prior)
        else:

            class ProjectedTransformer(nn.Module):
                def __init__(self, transformer):
                    super().__init__()
                    self.proj, self.transformer = nn.Linear(3, 192), transformer

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    if x.ndim == 2:
                        x = x.unsqueeze(0)  # add batch dimension: [seq, feat] -> [1, seq, feat]
                    # x is now [batch, seq, 3], proj expects last dim = 3
                    x = self.proj(x)  # [batch, seq, 192]
                    return self.transformer(x)

            trans_cfg = dict(
                vit=False,
                feature_space_dim=192,
                sequence_length=100,
                output_dim=8,
                num_layers=2,
                num_heads=12,
                head_dim=16,
                d_model=192,
            )

            base_trans = embedding_nets.TransformerEmbedding(trans_cfg)
            embedding_trans = ProjectedTransformer(base_trans)
            density_estimator_trans = posterior_nn(
                model="maf",
                embedding_net=embedding_trans,
                z_score_x="independent",
                z_score_y="independent",
            )

            inference = NPE(
                prior=prior,
                density_estimator=density_estimator_trans,
            )

            flatten_x = False  # pass full trajectories

    else:
        print("Using MAF")
        inference = NPE(prior=prior)
        flatten_x = True
        x_obs = x_obs.reshape(1, -1) 

    # ----------------------------------------------------------------
    # SNPE loop
    # ----------------------------------------------------------------
    proposal = prior
    posterior = None

    for r in range(num_rounds):

        print(f"\n========== SNPE ROUND {r+1}/{num_rounds} ==========")

        # Draw parameters from current proposal
        theta = proposal.sample((sims_per_round,))  # [N, 2]

        x = simulator_trajectory(theta, sim_cfg)    # [N, T, 3]

        if flatten_x:
            x = x.reshape(x.shape[0], -1)          # [N, T*3]

        density_estimator = (
            inference
            .append_simulations(theta, x, proposal=proposal)
            .train(
                validation_fraction=0.1,
                stop_after_epochs=20,
                max_num_epochs=300,
                show_train_summary=True,
            )
        )

        posterior = inference.build_posterior(
            density_estimator,
            sample_with="mcmc",
        )

        # Focus next round on observed x0
        proposal = posterior.set_default_x(x_obs)

        print("  round complete.")

    # ----------------------------------------------------------------
    # Posterior sampling at x_obs
    # ----------------------------------------------------------------
    print("\nSampling final posterior at observed x0...")

    samples = posterior.sample((5000,), x=x_obs)

    mean = samples.mean(dim=0)
    std  = samples.std(dim=0)

    print("\n==== FINAL SNPE RESULTS ====")
    print(f"True θ      : {true_theta}")
    print(f"Posterior μ : {mean}")
    print(f"Posterior σ : {std}")
    print(f"Error       : {mean - true_theta}")


if __name__ == "__main__":
    main()

