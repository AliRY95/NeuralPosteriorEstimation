# Neural Posterior Estimation for Cell Migration Analysis

A Python pipeline for inferring cell migration parameters using Neural Posterior Estimation (NPE) and Simulation-Based Inference (SBI). This project uses deep learning to estimate posterior distributions of model parameters given observed cell trajectory data.

## Overview

This repository implements a neural posterior estimation framework to infer biological parameters from cell migration trajectories. The pipeline supports:

- **Two field types**: Linear chemical gradients and cancer-cell-derived chemical fields
- **Two neural architectures**: Transformer-based embeddings and basic MAF (Masked Autoregressive Flow) on flattened sequences
- **Flexible parameter inference**: Estimate cell migration parameters (κ, d_τ) and chemical field properties

### Key Parameters

**Linear Field Mode** (default):
- `κ` (kappa): Chemotaxis strength
- `d_τ`: Persistence/turning rate

**Cancer Field Mode**:
- `κ` (kappa): Chemotaxis strength
- `d_τ`: Persistence/turning rate  
- `target_L`: Effective diffusion length in chemical field

## Installation

### Prerequisites
- Python 3.12+
- PyTorch
- SBI (Simulation-Based Inference)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/AliRY95/NeuralPosteriorEstimation.git
cd NeuralPosteriorEstimation
```

2. Install dependencies:
```bash
pip install -e .
```

### Required Dependencies
- `torch` - Deep learning framework
- `sbi` - Simulation-based inference library
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization

## Usage

### Basic Training

Train a neural posterior estimator with default settings:

```bash
python myNPE.py
```

This will:
1. Generate 10,000 simulated cell trajectories
2. Train an NPE model using MAF density estimator
3. Evaluate accuracy on 200 synthetic test cases
4. Save results to `data/accuracy_results.csv`

### Advanced Usage

```bash
# Train with more simulations
python myNPE.py -n 20000 -e 300

# Use transformer-based embedding for sequence processing
python myNPE.py -t

# Train with cancer field type
python myNPE.py -f cancer

# Load and evaluate a pre-trained model
python myNPE.py -l path/to/checkpoint.pt

# Save trained model checkpoint
python myNPE.py -s path/to/checkpoint.pt

# Combine options
python myNPE.py -n 15000 -e 300 -s ./models/posterior.pt -t -f cancer
```

### Command-Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--n-sim` | `-n` | 10000 | Number of simulations for training |
| `--eval-n` | `-e` | 200 | Number of test cases for evaluation |
| `--load` | `-l` | None | Path to existing posterior checkpoint |
| `--save` | `-s` | None | Path to save trained posterior |
| `--transformer` | `-t` | False | Use transformer embedding (default: basic MAF) |
| `--field-type` | `-f` | linear | Field type: `linear` or `cancer` |

## Project Structure

```
NeuralPosteriorEstimation/
├── myNPE.py                 # Main NPE training and inference script
├── snpe.py                  # Sequential NPE implementation
├── simulator.py             # Trajectory simulation wrapper
├── simulator_trajectory.py  # Population-level trajectory simulator
├── posterior_test.py        # Posterior evaluation and visualization
├── restartNetwork.py        # Model checkpointing utilities
├── plot_snp.py              # Plotting utilities
├── plot_trajectory.py       # Trajectory visualization
├── pyproject.toml           # Project configuration
├── data/                    # Output directory for results
├── MiscScripts/             # Utility scripts
└── Presentation/            # Presentation materials
```

## Core Components

### Simulator (`simulator.py`, `simulator_trajectory.py`)

Generates synthetic cell trajectories given model parameters:

```python
from simulator import simulator_trajectory

# Simulate trajectories for given parameters
thetas = prior.sample((1000,))  # [1000, n_params]
trajectories = simulator_trajectory(thetas, sim_cfg=sim_config)  # [1000, T_max, 3]
```

**Output format**: `[N, T, 3]` tensor with columns `[x(t), y(t), t_normalized]`

### Neural Posterior Estimator (`myNPE.py`)

Trains a conditional density estimator to approximate the posterior p(θ|x):

```python
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

# Create density estimator
density_estimator = posterior_nn(model="maf", z_score_x="independent")

# Train NPE
inference = NPE(prior=prior, density_estimator=density_estimator)
posterior = inference.append_simulations(thetas, x).train()

# Sample from posterior given observation
samples = posterior.sample(torch.Size([1000]), x=x_obs)
```

### Evaluation (`posterior_test.py`)

Evaluates posterior accuracy on synthetic test cases:

**Output metrics**:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **Bias** (systematic error)

**Generates**:
- CSV file with true vs estimated parameters
- PDF plots showing true vs estimated values with uncertainty bands

Example output:
```
Posterior accuracy on synthetic trajectories:
Test cases: 200

Param           MAE         RMSE         Bias
--------  -----------  -----------  -----------
0              0.1234       0.1567       0.0145
1              0.0089       0.0124      -0.0032
```

## Configuration

### Linear Field Configuration

```python
sim_cfg = dict(
    N_pop=1,              # Number of cell populations
    T_max=100.0,          # Simulation time
    field_type="linear",  # Linear chemical gradient
    s=1.0,                # Time step size
    theta_init=None,      # Initial parameter values (random if None)
    seed=123,             # Random seed for reproducibility
)

# Prior over (kappa, d_tau)
prior = BoxUniform(
    low=torch.tensor([1.0, 0.01]),
    high=torch.tensor([8.0, 0.15]),
)
```

### Cancer Field Configuration

```python
sim_cfg = dict(
    N_pop=1,
    T_max=400.0,
    field_type="cancer",
    target_Q=0.01,        # Source strength
    target_x0=1.0001,     # Source x-coordinate
    target_y0=1.0001,     # Source y-coordinate
    s=0.01,
    lambda_=1.0,          # Decay rate
    seed=123,
)

# Prior over (kappa, d_tau, target_L)
prior = BoxUniform(
    low=torch.tensor([1.0, 0.1, 4.0]),
    high=torch.tensor([4.0, 0.3, 8.0]),
)
```

## Output Files

### Results Directory (`data/`)

- **`accuracy_results.csv`**: Detailed results for each test case
  - Columns: `true_param_*`, `est_param_*`, `std_param_*`
  
- **`accuracy_plot_*.pdf`**: Visualization plots
  - True vs estimated parameter values
  - Uncertainty bands (±1 standard deviation)

### Model Checkpoints

Saved with `-s` flag:
```bash
python myNPE.py -s ./models/my_model.pt
```

Load checkpoint:
```bash
python myNPE.py -l ./models/my_model.pt
```

## Architecture Choices

### Basic MAF (Default)

- Flattens trajectory sequences to 1D vectors
- Masked Autoregressive Flow for density estimation
- Simpler, faster training
- Good for fixed-length trajectories

```bash
python myNPE.py  # Uses basic MAF
```

### Transformer-Based

- Preserves sequence structure with transformer embedding
- Projects 3D features to 192D, processes with multi-head attention
- Handles variable-length sequences better
- Slower but more flexible

```bash
python myNPE.py -t  # Uses transformer
```

## Examples

### Train and save a model

```bash
python myNPE.py -n 20000 -e 500 -s ./checkpoints/model_v1.pt
```

### Load model and evaluate on new data

```bash
python myNPE.py -l ./checkpoints/model_v1.pt -e 300
```

### Compare field types

```bash
# Linear field
python myNPE.py -n 10000 -f linear

# Cancer field
python myNPE.py -n 10000 -f cancer
```

### Use transformer for cancer field

```bash
python myNPE.py -t -f cancer -n 15000 -s ./models/cancer_transformer.pt
```

## Reproducibility

All scripts support seeding for reproducibility:

```python
seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
```

Results vary between runs due to:
- Neural network training randomness
- Stochastic sampling from posterior
- Random trajectory simulation

To ensure reproducibility:
1. Set `seed` parameter in configuration
2. Use same number of simulations (`-n` flag)
3. Use same model architecture (`-t` for transformer)
4. Load pre-trained checkpoint instead of training

## Mathematical Background

### Simulation Model

Cell trajectories follow a stochastic model combining:

1. **Chemotaxis**: Migration along chemical gradient
   - Strength controlled by κ (kappa)
   
2. **Persistence**: Memory of previous direction
   - Controlled by d_τ
   
3. **Chemical Field**: Source-dependent gradient
   - Linear: Constant directional preference
   - Cancer: Radial symmetric diffusion from point sources

### Posterior Estimation

Using Neural Posterior Estimation:
- Train a neural density estimator q(θ|x) to approximate p(θ|x)
- Condition on observed trajectory x_obs
- Sample parameter estimates from posterior distribution

## Performance Notes

- **Training time**: ~1-10 minutes (depends on `-n` and architecture)
- **GPU recommended**: Uses CUDA if available
- **Memory**: ~2-4GB for typical configurations
- **Accuracy**: Improves with more simulations (diminishing returns after 20k)

## Troubleshooting

### Out of Memory
Reduce number of simulations:
```bash
python myNPE.py -n 5000
```

### Slow training
Use basic MAF (default) instead of transformer:
```bash
python myNPE.py  # Faster than -t
```

### Import errors
Ensure all dependencies are installed:
```bash
pip install torch sbi numpy matplotlib
```

## Citation

If you use this code, please cite:

```bibtex
@software{NeuralPosteriorEstimation2025,
  author = {Ali R. Y.},
  title = {Neural Posterior Estimation for Cell Migration Analysis},
  year = {2025},
  url = {https://github.com/AliRY95/NeuralPosteriorEstimation}
}
```

## References

- **SBI Framework**: Cranmer et al., "Simulation-based inference using neural posterior estimation" (2020)
- **Neural Density Estimation**: Papamakarios et al., "Masked Autoregressive Flow for Density Estimation" (2017)
- **Transformers**: Vaswani et al., "Attention is All You Need" (2017)

## License

This project is provided as-is for research purposes.

## Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/AliRY95/NeuralPosteriorEstimation).

---

**Last Updated**: December 2025
**Python Version**: 3.12+
