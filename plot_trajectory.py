from pathlib import Path

from simulator_trajectory import (
    plot_field,
    plot_segment_angle_distributions,
    plot_trajectories,
    simulate_population,
)


def main():
    # Define parameters
    theta = {
        "kappa": 8.,  # von Mises concentration
        "d_tau": 0.25,  # chemo sensitivity
        "target_L": 1.,  # chemo decay length scale 
    }

    config = {
        "N_pop": 16,  # number of walkers
        "T_max": 100.0,  # total time horizon
        "field_type": "cancer",  # chemo-attractant field type
        "target_Q": 1.0,  # target strength
        "target_x0": 10.0,  # target x-coordinate
        "target_y0": 10.0,  # target y-coordinate
        "lambda": 0.5,  # Poisson turn rate
        "s": 1.0,  # speed
        "seed": 1234,  # base seed for reproducible diversity
    }

    # Plot the field
    plot_field(field_type=config["field_type"], theta=theta, config=config, show=False)
    
    # Simulate and plot
    trajectories = simulate_population(theta=theta, config=config)
    # Save overlay plot of trajectories (no individual plots)
    plot_trajectories(trajectories, overlay=True, seed=config.get("seed"), theta=theta)
    # Save circular (rose) histograms of segment angles in a grid (single file)
    plot_segment_angle_distributions(
        trajectories, overlay=True, bins=72, seed=config.get("seed"), degrees=True, theta=theta
    )

    plot_dir = Path(__file__).parent / "plot"
    print(f"Saved plots in: {plot_dir.resolve()}")


if __name__ == "__main__":
    main()
