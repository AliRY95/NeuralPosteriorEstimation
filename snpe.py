import numpy as np
import torch
from sbi.utils import BoxUniform
from sbi.inference import NPE

from simulator_trajectory import *
from new_sum import population_simple_summary, simulator


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

#SNPE

num_rounds = 4
sims_per_round = 2500

inference = NPE(prior)

proposal = prior

for r in range(num_rounds):

    print(f"\n========== SNPE ROUND {r+1}/{num_rounds} ==========")

    # sample parameters from proposal
    theta = proposal.sample((sims_per_round,))

    # simulate 
    x = simulator(theta, sim_cfg)

    # train npe
    density_estimator = (
        inference
        .append_simulations(theta, x, proposal=proposal)
        .train()
    )

    # posterior = inference.build_posterior(density_estimator)
    posterior = inference.build_posterior(
    density_estimator,
    sample_with="mcmc"
)

    # update proposal 
    proposal = posterior.set_default_x(x_obs)


posterior_snpe = posterior

samples = posterior_snpe.sample((5000,), x=x_obs)

print("\nPosterior mean:", samples.mean(dim=0))
print("Posterior std :", samples.std(dim=0))


# Testing
N_test = 100

# draw test parameters from prior
theta_test = prior.sample((N_test,))

errors = []
biases = []

for true_theta in theta_test:
    true_kappa = float(true_theta[0])
    true_dtau  = float(true_theta[1])
    theta_true = {
        "kappa": true_kappa,
        "d_tau": true_dtau,
    }
    x_obs = torch.tensor(
        population_simple_summary(theta_true, sim_cfg)
    )
    samples = posterior_snpe.sample((100,), x=x_obs)

    est = samples.mean(dim=0)
    errors.append(est - true_theta)
    biases.append(est)

errors = torch.stack(errors)
biases = torch.stack(biases)

mae  = errors.abs().mean(dim=0)
rmse = torch.sqrt((errors**2).mean(dim=0))
bias = errors.mean(dim=0)

print("\nPosterior accuracy over test set:")
print("Metric      kappa       d_tau")
print("--------------------------------")
print(f"MAE      {mae[0]:10.4f}  {mae[1]:10.4f}")
print(f"RMSE     {rmse[0]:10.4f}  {rmse[1]:10.4f}")
print(f"Bias     {bias[0]:10.4f}  {bias[1]:10.4f}")

