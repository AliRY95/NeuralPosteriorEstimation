from pathlib import Path

from simulator_trajectory import (
    plot_segment_angle_distributions,
    plot_trajectories,
    simulate_population,
)


def main():
    # Define parameters
    theta = {
        "kappa": 50.0,  # von Mises concentration
        "d_tau": 0.25,  # chemo sensitivity
    }

    config = {
        "N_pop": 128,  # number of walkers
        "T_max": 80.0,  # total time horizon
        "lambda": 0.6,  # turn rate
        "s": 1.0,  # speed
        "seed": 123,  # base seed for reproducible diversity
    }

    # Simulate and plot
    trajectories = simulate_population(theta=theta, config=config)
    # Save overlay plot of trajectories (no individual plots)
    plot_trajectories(trajectories, overlay=True, seed=config.get("seed"), theta=theta)
    # Save circular (rose) histograms of segment angles in a grid (single file)
    plot_segment_angle_distributions(
        trajectories, overlay=True, bins=36, seed=config.get("seed"), degrees=True, theta=theta
    )

    plot_dir = Path(__file__).parent / "plot"
    print(f"Saved plots in: {plot_dir.resolve()}")


if __name__ == "__main__":
    main()
