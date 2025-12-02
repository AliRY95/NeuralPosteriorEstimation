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
from misc import load_posterior, save_posterior, save_posterior_transformer


def main():
    parser = argparse.ArgumentParser(description="Train or load posterior and evaluate.")
    parser.add_argument("--n-sim", type=int, default=10000, help="Number of simulations for training if not loading")
    parser.add_argument("--eval-n", type=int, default=200, help="Number of synthetic test draws for accuracy evaluation")
    parser.add_argument("--load", type=str, default=None, help="Path to existing posterior checkpoint to load instead of training")
    parser.add_argument("--save", type=str, default=None, help="Path (file or dir) to save trained posterior checkpoint")
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
    num_simulations = args.n_sim
    thetas = prior.sample((num_simulations,))
    xs_xy_seq = simulator_trajectory(thetas, sim_cfg=sim_cfg)
    # xs_xy_flat = xs_xy_seq.reshape(xs_xy_seq.shape[0], -1)
    
    # if args.load:
    #     posterior_npe = load_posterior(args.load, prior)
    # else:
    #     density_estimator_npe = posterior_nn(
    #         model='maf',
    #         z_score_x='none',
    #         z_score_y='none',
    #     )
    #     inference_npe = NPE(prior=prior, density_estimator=density_estimator_npe)
    #     posterior_npe = inference_npe.append_simulations(thetas, xs_xy_seq).train()
    #     if args.save:
    #         save_posterior(args.save, prior, posterior_npe, input_length=xs_xy_seq.shape[1])
    # # Evaluation
    # eval_accuracy_trajectory(args.eval_n, sim_cfg, prior, posterior_npe, use_xy=True)
    
    # -----------------------------
    # Using TransformerEmbedding
    # -----------------------------
    if args.load:
        posterior_trans = load_posterior(args.load, prior)
    else:
        class ProjectedTransformer(nn.Module):
            def __init__(self, transformer):
                super().__init__()
                self.proj, self.transformer = nn.Linear(2, 192), transformer

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.ndim == 2:
                    x = x.unsqueeze(-1)
                x = self.proj(x)
                return self.transformer(x)
        

        trans_cfg = dict(
            vit=False,             # Use standard transformer, not ViT-style
            feature_space_dim=192, # d_model = num_heads * head_dim
            sequence_length=100,   # Number of time points
            output_dim=8,          # Smaller output feature dimension
            num_layers=2,          # Lighter depth
            num_heads=12,          # Keep default heads to match sbi internals
            head_dim=16,           # 12 * 16 = 192
            d_model=192,           # same as feature_space_dim
        )

        base_trans = embedding_nets.TransformerEmbedding(trans_cfg)
        embedding_trans = ProjectedTransformer(base_trans)
        density_estimator_trans = posterior_nn(
            model="maf",
            embedding_net=embedding_trans,
            z_score_x="none",
            z_score_y="none",
        )
        inference_trans = NPE(prior=prior, density_estimator=density_estimator_trans)
        posterior_trans = inference_trans.append_simulations(thetas, xs_xy_seq).train()
        if args.save:
            # Flattened XY length is 2 * T
            flat_len = xs_xy_seq.shape[1] * xs_xy_seq.shape[2]
            save_posterior_transformer(args.save, prior, posterior_trans, embedding_trans, trans_cfg, input_length=flat_len)
    # Evaluation
    eval_accuracy_trajectory(args.eval_n, sim_cfg, prior, posterior_trans, use_xy=True, flatten_xy=False)


if __name__ == "__main__":
    main()
