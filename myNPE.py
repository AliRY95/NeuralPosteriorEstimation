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
    parser = argparse.ArgumentParser(description="Train or load posterior and evaluate.")
    parser.add_argument(
        "-n",
        "--n-sim",
        type=int,
        default=10000,
        help="Number of simulations for training if not loading",
    )
    parser.add_argument(
        "-e",
        "--eval-n",
        type=int,
        default=200,
        help="Number of synthetic test draws for accuracy evaluation",
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
    x = simulator_trajectory(thetas, sim_cfg=sim_cfg)

    # -----------------------------
    # Architecture selection
    # -----------------------------
    # Transformer path
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
            inference_trans = NPE(prior=prior, density_estimator=density_estimator_trans)
            posterior = inference_trans.append_simulations(thetas, x).train()
            if args.save:
                flat_len = x.shape[1] * x.shape[2]
                save_posterior_transformer(
                    args.save, prior, posterior, embedding_trans, trans_cfg, input_length=flat_len
                )
        # Evaluation (transformer uses sequences, not flattened)
        eval_accuracy_trajectory(args.eval_n, sim_cfg, prior, posterior, flatten=False)

    # Basic MAF on flattened XY
    else:
        if args.load:
            posterior = load_posterior(args.load, prior)
        else:
            # Flatten XY sequences for MAF
            xs_xy_flat = x.reshape(x.shape[0], -1)

            density_estimator = posterior_nn(
                model="maf",
                z_score_x="independent",
                z_score_y="independent",
            )
            inference = NPE(prior=prior, density_estimator=density_estimator)
            tmp = inference.append_simulations(thetas, xs_xy_flat).train()
            posterior = inference.build_posterior(tmp, sample_with="direct")
            if args.save:
                save_posterior(args.save, prior, posterior, input_length=xs_xy_flat.shape[1])
        # Evaluation (basic MAF uses flattened trajectories)
        eval_accuracy_trajectory(args.eval_n, sim_cfg, prior, posterior, flatten=True)


if __name__ == "__main__":
    main()
