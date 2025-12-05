import numpy as np
import torch
from pathlib import Path
from sbi.inference import NPE
from sbi.utils import BoxUniform
import csv

from simulator_trajectory import *

def simulator(theta_batch: torch.Tensor, config: dict) -> torch.Tensor:
    xs = []

    theta_np = theta_batch.cpu().numpy()

    for kappa, d_tau in theta_np:
        theta_dict = {
            "kappa": float(kappa),
            "d_tau": float(d_tau),
        }

        summary = population_simple_summary(
            theta=theta_dict,
            config=config,
        )

        xs.append(torch.tensor(summary, dtype=torch.float32))

    return torch.stack(xs)

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

prior = BoxUniform(
    low=torch.tensor([1.0, 0.01]),
    high=torch.tensor([8.0, 0.15]),
)

sim_cfg = dict(
    N_pop=1,
    T_max=100.0,
    lambdaa=0.5,
    s=1.0,
    theta_init=None,
    seed=seed,
)


true_kappa = 3.0
true_dtau = 0.12

theta_true = {"kappa": true_kappa, "d_tau": true_dtau}

x_obs = torch.tensor(
    population_simple_summary(theta_true, sim_cfg),
    dtype=torch.float32,
)

print("Observed summary:", x_obs)

# SNPE
num_rounds = 4
sims_per_round = 1000

inference = NPE(prior)
proposal = prior

# Output directory for per-round dumps
out_dir = (Path(__file__).resolve().parent / "sbi-logs" / "SNPE").resolve()
out_dir.mkdir(parents=True, exist_ok=True)

for r in range(num_rounds):
    print(f"\n========== SNPE ROUND {r+1}/{num_rounds} ==========")

    # sample parameters from proposal
    theta = proposal.sample((sims_per_round,))

    # simulate
    x = simulator(theta, sim_cfg)

    # train npe
    density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()

    # posterior = inference.build_posterior(density_estimator)
    posterior = inference.build_posterior(density_estimator, sample_with="mcmc")

    # update proposal
    proposal = posterior.set_default_x(x_obs)


posterior_snpe = posterior

samples = posterior_snpe.sample((10,), x=x_obs)

print("\nPosterior mean:", samples.mean(dim=0))
print("Posterior std :", samples.std(dim=0))


# Testing
sim_cfg["seed"] = seed + 1  # different seed for testing
N_test = 10

# draw test parameters from prior
theta_test = prior.sample((N_test,))

errors = []
biases = []

for true_theta in theta_test:
    true_kappa = float(true_theta[0])
    true_dtau = float(true_theta[1])
    theta_true = {
        "kappa": true_kappa,
        "d_tau": true_dtau,
    }
    x_obs = torch.tensor(population_simple_summary(theta_true, sim_cfg), dtype=torch.float32)
    samples = posterior_snpe.sample((100,), x=x_obs)

    est = samples.mean(dim=0)
    errors.append(est - true_theta)
    biases.append(est)

errors = torch.stack(errors)
biases = torch.stack(biases)

mae = errors.abs().mean(dim=0)
rmse = torch.sqrt((errors**2).mean(dim=0))
bias = errors.mean(dim=0)

print("\nPosterior accuracy over test set:")
print("Metric      kappa       d_tau")
print("--------------------------------")
print(f"MAE      {mae[0]:10.4f}  {mae[1]:10.4f}")
print(f"RMSE     {rmse[0]:10.4f}  {rmse[1]:10.4f}")
print(f"Bias     {bias[0]:10.4f}  {bias[1]:10.4f}")
