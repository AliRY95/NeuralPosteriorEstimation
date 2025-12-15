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
        "kappa": 4.,  # von Mises concentration
        "d_tau": 0.2,  # chemo sensitivity
        "target_L": 8,  # chemo decay length scale 
    }

    config = {
        "N_pop": 1,  # number of walkers
        "T_max": 300.0,  # total time horizon
        "field_type": "cancer",  # chemo-attractant field type
        "target_Q": .01,  # target strength
        "target_x0": 1.0001,  # target x-coordinate
        "target_y0": 1.0001,  # target y-coordinate
        "lambda_": 1.,  # Poisson turn rate
        "s": .01,  # speed
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
