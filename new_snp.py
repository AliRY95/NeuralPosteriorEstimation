#!/usr/bin/env python3

import argparse
import numpy as np
import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt

from sbi.inference import NPE
from sbi.neural_nets import embedding_nets, posterior_nn
from sbi.utils import BoxUniform

from restartNetwork import (
    load_posterior_transformer,
    save_posterior_transformer,
)
from simulator import simulator_trajectory
from plot_snp import *

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
        help="Path to existing posterior checkpoint to load instead of training (transformer only).",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        default=None,
        help="Path (file or dir) to save trained posterior checkpoint (transformer only).",
    )
    parser.add_argument(
        "-t",
        "--transformer",
        action="store_true",
        help="Use transformer embedding; default is basic MAF on flattened trajectories",
    )
    parser.add_argument(
        "-r",
        "--rounds",
        type=int,
        default=4,
        help="Number of SNPE rounds",
    )
    parser.add_argument(
        "-f",
        "--field-type",
        type=str,
        choices=["Linear", "Cancer"],
        default="Linear",
        help="Field type for simulation: 'linear' (default) or 'cancer'",
    )

    args = parser.parse_args()

    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ----------------------------------------------------------------
    # Prior 
    # ----------------------------------------------------------------
    if args.field_type == "Cancer":
        # Prior over (kappa, d_tau, target_L)
        theta_min = torch.tensor([1.0, 0.1, 4.0])
        theta_max = torch.tensor([4.0, 0.3, 8.0])
        prior = BoxUniform(theta_min, theta_max)

        sim_cfg = dict(
            N_pop=1,
            T_max=400.0,
            field_type="cancer",
            target_Q=0.01,
            target_x0=1.0001,
            target_y0=1.0001,
            s=0.01,
            lambda_=1.0,          
            theta_init=None,
            seed=seed,
        )

        true_theta = torch.tensor([3.0, 0.2, 6.0])

    else:
        # Prior over (kappa, d_tau)
        theta_min = torch.tensor([1.0, 0.01])
        theta_max = torch.tensor([8.0, 0.15])
        prior = BoxUniform(theta_min, theta_max)

        sim_cfg = dict(
            N_pop=1,
            T_max=100.0,
            field_type="linear",
            s=1.0,
            # lambda_=0.5,         
            theta_init=None,
            seed=seed,
        )

        true_theta = torch.tensor([3.0, 0.12])


    x_obs = simulator_trajectory(true_theta.unsqueeze(0), sim_cfg)

    # ----------------------------------------------------------------
    # SNPE setup
    # ----------------------------------------------------------------
    num_rounds = args.rounds
    sims_per_round = args.n_sim // num_rounds

    print(f"\nSNPE SETTINGS:")
    print(f"  field_type        = {args.field_type}")
    print(f"  transformer       = {args.transformer}")
    print(f"  rounds            = {num_rounds}")
    print(f"  sims_per_round    = {sims_per_round}")
    print(f"  total simulations = {sims_per_round * num_rounds}")
    print("-" * 60)

    # ----------------------------------------------------------------
    # Embedding 
    # ----------------------------------------------------------------
    embedding_trans = None
    trans_cfg = None

    if args.transformer:
        if args.load:
            posterior = load_posterior_transformer(args.load, prior)
            inference = None
            flatten_x = False
        else:

            class ProjectedTransformer(nn.Module):
                def __init__(self, transformer, d_model=192):
                    super().__init__()
                    self.proj = nn.Linear(3, d_model)
                    self.transformer = transformer

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    if x.ndim == 2:
                        x = x.unsqueeze(0)   
                    x = self.proj(x)       
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
            embedding_trans = ProjectedTransformer(base_trans, d_model=192)

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

            flatten_x = False  

    else:
        print("Using MAF (flattened)")
        inference = NPE(prior=prior)
        flatten_x = True
        x_obs = x_obs.reshape(1, -1)

    # ----------------------------------------------------------------
    # SNPE loop 
    # ----------------------------------------------------------------
    proposal = prior

    if args.load and args.transformer:
        proposal = posterior.set_default_x(x_obs)
    else:
        posterior = None

        for r in range(num_rounds):
            print(f"\n========== SNPE ROUND {r+1}/{num_rounds} ==========")
            theta = proposal.sample((sims_per_round,))

            x = simulator_trajectory(theta, sim_cfg)  

            if flatten_x:
                x = x.reshape(x.shape[0], -1)         

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

            proposal = posterior.set_default_x(x_obs)

            print("  round complete.")

        if args.save and args.transformer:
            flat_len = int(x_obs.shape[1] * x_obs.shape[2]) if x_obs.ndim == 3 else int(x_obs.shape[1])
            save_posterior_transformer(
                args.save,
                prior,
                posterior,
                embedding_trans,
                trans_cfg,
                input_length=flat_len,
                field_type=args.field_type,
            )

    # ----------------------------------------------------------------
    # Posterior sampling at x_obs 
    # ----------------------------------------------------------------
    print("\nSampling final posterior at observed x_obs...")

    samples = posterior.sample((5000,), x=x_obs)

    mean = samples.mean(dim=0)
    std = samples.std(dim=0)

    print("\n==== FINAL SNPE RESULTS ====")
    print(f"True θ      : {true_theta}")
    print(f"Posterior μ : {mean}")
    print(f"Posterior σ : {std}")
    print(f"Error       : {mean - true_theta}")
    
    results = local_inverse_trajectory_test(
    posterior,
    simulator_trajectory,
    theta_star=true_theta,
    prior=prior,
    sim_cfg=sim_cfg,
    x_obs=x_obs,
    flatten_x=flatten_x,
    n_tests=3,
    n_post_samples=3000,
    )

    plot_dir = Path(__file__).parent / "data"
    plot_dir.mkdir(exist_ok=True)

    plot_local_inverse_results(
        results,
        field_type=args.field_type,
        out_path=plot_dir / f"local_inverse_trajs_{args.field_type}.pdf",
    )


if __name__ == "__main__":
    main()





