
import numpy as np
import torch
import matplotlib.pyplot as plt

from sbi.utils import BoxUniform
from sbi.inference import NPE

from simulator_trajectory import *


def population_simple_summary(
    theta: dict,   # {"kappa", "d_tau"}
    config: dict,  
) -> np.ndarray:
    
    # simulate 
    trajectories = simulate_population(theta=theta, config=config)

    pull_x = []
    cos_turns = []

    for tr in trajectories:

        # avg pull along chem grad (to the rigjt)
        pull_x.append(float(tr.x[-1] - tr.x[0]))

        # turning angles
        angles = tr.segment_angles
        if angles.size > 1:
            dtheta = np.diff(np.unwrap(angles))
            cos_turns.extend(np.cos(dtheta))

    mean_pull_x = float(np.mean(pull_x))
    mean_cos_turn = float(np.mean(cos_turns)) if cos_turns else 0.0

    return np.array([mean_pull_x, mean_cos_turn], dtype=np.float32)


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
    high=torch.tensor([8.0, 0.15])
)

#### simulate

sim_cfg = dict(
    N_pop=1,
    T_max=100.0,
    lambdaa=0.5,
    s=1.0,
    theta_init=None,
    seed=seed,
)

num_simulations = 10_000

theta_train = prior.sample((num_simulations,))

x_train = simulator(theta_train, sim_cfg)

# train npe

inference = NPE(prior)
density_estimator = (
    inference
        .append_simulations(theta_train, x_train)
        .train()
)

posterior = inference.build_posterior()


### inference on observed data
true_kappa = 3.0
true_dtau  = 0.12

theta_true = {
    "kappa": true_kappa,
    "d_tau": true_dtau,
}


x_obs = torch.tensor(
    population_simple_summary(theta_true, sim_cfg)
)

samples = posterior.sample((5000,), x=x_obs)

print("Posterior mean:", samples.mean(dim=0))
print("Posterior std :", samples.std(dim=0))



# test time :)

N_test = 500

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
    samples = posterior.sample((2000,), x=x_obs)

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

