from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import kv

Field = Callable[[float, float], float]
Gradient = Callable[[float, float], tuple[float, float]]


################################################################################
# Trajectory class
################################################################################
@dataclass(slots=True)
class Trajectory:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    segment_angles: np.ndarray

    @staticmethod
    def _segment_angles(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if x.size < 2:
            return np.empty(0)
        dx = np.diff(x)
        dy = np.diff(y)
        return np.arctan2(dy, dx)

    def interpolate_to_integers(self, T_max: float) -> Trajectory:
        M = int(np.floor(T_max))
        t_int = np.arange(1.0, M + 1, 1.0)
        x_int = np.empty_like(t_int)
        y_int = np.empty_like(t_int)

        j = 0
        for i, ti in enumerate(t_int):
            while j < self.t.size - 1 and self.t[j + 1] < ti:
                j += 1

            if j >= self.t.size - 1:
                x_int[i] = self.x[-1]
                y_int[i] = self.y[-1]
            else:
                t0, t1 = self.t[j], self.t[j + 1]
                x0, x1 = self.x[j], self.x[j + 1]
                y0, y1 = self.y[j], self.y[j + 1]
                alpha = (ti - t0) / (t1 - t0)
                x_int[i] = (1.0 - alpha) * x0 + alpha * x1
                y_int[i] = (1.0 - alpha) * y0 + alpha * y1

        angles_int = self._segment_angles(x_int, y_int)
        return Trajectory(t=t_int, x=x_int, y=y_int, segment_angles=angles_int)


################################################################################
# Miscellaneous functions
################################################################################
# Default chemo-attractant field and gradient
def default_field(x: float, y: float) -> float:
    """Simple linear chemo-attractant field C(x, y) = 0.01 x."""
    return 0.01 * x

# Cancer-like Gaussian chemo-attractant field
# This is origin-centered; modify by shifting the input coordinates (x, y) to change the center.
def cancer_field(x: float, y: float, x0: float, y0: float, Q: float, lmbda: float) -> float:
    """
    Chemo-attractant field with a peak at (x0, y0).
    Uses modified Bessel function K0 which decreases with distance.
    Lambda controls the decay rate - larger lambda means faster decay.
    """
    r_squared = (x - x0)**2 + (y - y0)**2
    # Avoid singularity at target location
    if r_squared < 1e-5:
        return Q / (2.0 * np.pi)  # Maximum concentration at target
    
    r = np.sqrt(r_squared)
    # For K0, we want argument to grow with distance for proper decay
    # Using sqrt(lambda) * r so larger lambda → faster decay
    arg = np.sqrt(lmbda) * r
    
    # K0 diverges at 0 and decays exponentially for large arguments
    # This gives us a peak at the target and decay away from it
    return Q / (2.0 * np.pi) * kv(0, arg)


# Cancer-like Gaussian chemo-attractant field gradient
# This is origin-centered; modify by shifting the input coordinates (x, y) to change the center.
def cancer_field_gradient(x: float, y: float, x0: float, y0: float, Q: float, lmbda: float) -> tuple[float, float]:
    """
    Gradient of chemo-attractant field with a peak at (x0, y0).
    Uses modified Bessel function K0 which decreases with distance.
    Lambda controls the decay rate - larger lambda means faster decay.
    """
    r_squared = (x - x0)**2 + (y - y0)**2
    r = np.sqrt(r_squared)
    arg = np.sqrt(lmbda) * r
    grad = - Q / (2.0 * np.pi) * np.sqrt(lmbda) * kv(1, arg)
    # Gradient components
    return grad * (x - x0) / r, grad * (y - y0) / r


# Central finite-difference approximation of gradient
def central_gradient(C: Field, x: float, y: float, h: float = 1e-3) -> tuple[float, float]:
    """Central finite-difference approximation of ∇C."""
    dCdx = (C(x + h, y) - C(x - h, y)) / (2.0 * h)
    dCdy = (C(x, y + h) - C(x, y - h)) / (2.0 * h)
    return dCdx, dCdy


################################################################################
# Simulation functions
################################################################################
# Codling-style correlated random walk simulator
def simulate_codling_walk(
    theta: dict,
    config: dict,
    rng: np.random.Generator | None = None,
) -> Trajectory:
    """
    theta keys:
    - kappa (float): concentration parameter of von Mises turn angle distribution
    - d_tau (float): chemo sensitivity parameter for biasing turn angles
    - target_L (float): chemo decay length scale for cancer field

    config keys:
    - T_max (float): total simulation horizon (time units). Default: 100.0
    - field_type (str): chemo-attractant field type. Default: linear field 0.01·x
    - target_Q (float): chemo source strength for cancer field
    - target_x0 (float): chemo source x-coordinate for cancer field
    - target_y0 (float): chemo source y-coordinate for cancer field
    - s (float): speed (distance units per time unit). Default: 1.0
    - lambda (float): Poisson turn rate λ (events per unit time). Default: 0.5
    - theta_init (float|None): initial heading angle in radians; if None, draws U[0, 2π). Default: None
    - seed (int|None): RNG seed for reproducible trajectory. Default: None
    """
    # extract parameters from dicts
    kappa = float(theta.get("kappa"))
    d_tau = float(theta.get("d_tau"))

    s = float(config.get("s", 1.0))
    lambda_ = float(config.get("lambda", 0.5))
    T_max = float(config.get("T_max", 100.0))
    field_type = config.get("field_type", "linear")
    theta_init = config.get("theta_init", None)
    # initialize local RNG (avoid global state)
    if rng is None:
        seed = config.get("seed")
        try:
            rng = np.random.default_rng(None if seed is None else int(seed))
        except Exception:
            rng = np.random.default_rng()

    
    if field_type == "cancer":
        L = 10**(-float(theta.get("target_L")))
        Q = float(config.get("target_Q"))
        x0 = float(config.get("target_x0"))
        y0 = float(config.get("target_y0"))
        C = lambda x, y: cancer_field(x, y, x0, y0, Q, L)
        assert Q > 0.0 and L > 0.0, "Cancer field parameters must be positive."
        gradC = lambda x, y: cancer_field_gradient(x, y, x0, y0, Q, L)
        max_grad = Q / (2.0 * np.pi) * np.sqrt(L) * kv(1, 1.e-4)
    else:
        C = default_field
        gradC = lambda x, y, h=1e-5: central_gradient(C, x, y, h)
        max_grad = 0.01


    t = 0.0
    x = 0.0
    y = 0.0
    theta = theta_init if theta_init is not None else rng.uniform(0.0, 2.0 * np.pi)

    t_raw = [t]
    x_raw = [x]
    y_raw = [y]
    segment_angles = []

    while t < T_max:
        dt = rng.exponential(1.0 / lambda_)

        # Compute gradient for chemotaxis and adaptive speed
        dCdx, dCdy = gradC(x, y)
        grad_mag = np.sqrt(dCdx**2 + dCdy**2)
        
        # Adaptive speed: slower when gradient is high (near target)
        # Speed decreases as gradient increases beyond threshold
        if grad_mag > max_grad:
            current_speed = 0.0
        else:
            current_speed = np.exp(-grad_mag / max_grad) * s
        
        if dCdx == 0.0 and dCdy == 0.0:
            theta_pref = theta
        else:
            theta_pref = np.arctan2(dCdy, dCdx)

        mu = -d_tau * (theta - theta_pref)
        delta_theta = rng.vonmises(mu, kappa)
        theta += delta_theta

        x_new = x + current_speed * dt * np.cos(theta)
        y_new = y + current_speed * dt * np.sin(theta)
        segment_angles.append(np.arctan2(y_new - y, x_new - x))
        x, y = x_new, y_new
        t += dt

        t_raw.append(t)
        x_raw.append(x)
        y_raw.append(y)

    return Trajectory(
        t=np.array(t_raw),
        x=np.array(x_raw),
        y=np.array(y_raw),
        segment_angles=np.array(segment_angles),
    )


# Simulate a population of walkers using theta/config dictionaries.
def simulate_population(
    theta: dict,  
    config: dict,
    rng: np.random.Generator | None = None,
) -> Sequence[Trajectory]:
    """
    theta keys:
    - kappa (float): concentration parameter of von Mises turn angle distribution
    - d_tau (float): chemo sensitivity parameter for biasing turn angles
    - target_Q (float): chemo source strength for cancer field
    - target_D (float): chemo diffusion coefficient for cancer field
    - target_L (float): chemo decay length scale for cancer field

    config keys:
    - N_pop (int): number of walkers. Default: 1
    - T_max (float): total simulation horizon (time units). Default: 100.0
    - field_type (str): chemo-attractant field type. Default: linear field 0.01·x
    - s (float): constant speed (distance units per time unit). Default: 1.0
    - lambda (float): Poisson turn rate λ (events per unit time). Default: 0.5
    - theta_init (float|None): initial heading angle in radians; if None, draws U[0, 2π). Default: None
    - seed (int|None): RNG seed for reproducible trajectory. Default: None
    """
    # extract N_pop from config
    N_pop = int(config.get("N_pop", 1))

    trajectories = []
    base_seed = config.get("seed")
    # If an RNG is provided, we can spawn independent streams by varying seed; else create per-walker RNGs.
    for i in range(N_pop):
        local_config = dict(config)
        local_rng: np.random.Generator | None
        if rng is not None:
            # derive a new RNG using an offset seed if base_seed exists; otherwise, use a fresh rng
            if base_seed is not None:
                try:
                    local_rng = np.random.default_rng(int(base_seed) + i)
                except Exception:
                    local_rng = np.random.default_rng()
            else:
                local_rng = np.random.default_rng()
        else:
            # create a dedicated RNG per walker (reproducible diversity)
            try:
                local_rng = np.random.default_rng(None if base_seed is None else int(base_seed) + i)
            except Exception:
                local_rng = np.random.default_rng()
        traj = simulate_codling_walk(theta=theta, config=local_config, rng=local_rng)
        trajectories.append(traj)

    return trajectories

def population_simple_summary(
    theta: dict,
    config: dict,
) -> np.ndarray:
    """
    Compute a simple 2-dimensional summary statistic for a population of walkers.
    
    Returns a summary vector of shape (2,) containing:
    1. Mean displacement along the x-axis (chemotactic gradient direction)
    2. Mean cosine of turn angles (indicator of directional persistence)
    """
    # Simulate population of trajectories
    trajectories = simulate_population(theta=theta, config=config)

    pull_x = []  # Net displacement along chemotactic gradient (x-axis)
    cos_turns = []  # Cosine of consecutive turn angles

    for traj in trajectories:
        # Measure net displacement along x-axis (chemotactic gradient direction)
        x_displacement = float(traj.x[-1] - traj.x[0])
        pull_x.append(x_displacement)

        # Compute turn angle statistics
        angles = traj.segment_angles
        if angles.size > 1:
            # Unwrap angles to avoid discontinuities at ±π, then compute differences
            dtheta = np.diff(np.unwrap(angles))
            # Cosine of turn angles: values near 1 indicate straighter paths
            cos_turns.extend(np.cos(dtheta))

    # Aggregate over population
    mean_pull_x = float(np.mean(pull_x))
    mean_cos_turn = float(np.mean(cos_turns)) if cos_turns else 0.0

    return np.array([mean_pull_x, mean_cos_turn], dtype=np.float32)


################################################################################
# Plot
################################################################################
# Plot trajectories
def plot_trajectories(
    trajectories: Trajectory | Sequence[Trajectory],
    *,
    overlay: bool = True,
    out_dir: str | None = None,
    seed: int | None = None,
    show: bool = False,
    fname_prefix: str = "trajectory",
    theta: dict | None = None,
) -> None:
    """
    Plot a single trajectory or a sequence of trajectories.

    Parameters:
    - trajectories: Trajectory or list/sequence of Trajectory
    - overlay: if True and `trajectories` is a sequence, overlay all on one figure;
               if False, save one file per trajectory.
    - out_dir: directory to save figures; defaults to a 'plot' folder next to this file.
    - seed: optional seed annotation to include in filenames.
    - show: whether to display figures via plt.show() (best-effort).
    - fname_prefix: filename prefix for saved figures.
    """
    try:
        from pathlib import Path

        import matplotlib.pyplot as plt
    except Exception:
        return

    # Normalize input to list
    if isinstance(trajectories, Trajectory):
        traj_list = [trajectories]
    else:
        traj_list = list(trajectories)

    plot_dir = Path(out_dir) if out_dir is not None else Path(__file__).parent / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)
    seed_str = f"seed_{int(seed)}" if seed is not None else "noseed"
    theta_str = ""
    if isinstance(theta, dict):
        try:
            kappa_val = float(theta.get("kappa"))
            dtau_val = float(theta.get("d_tau"))
            theta_str = f"_kappa{kappa_val:.3g}_dtau{dtau_val:.3g}"
        except Exception:
            pass

    if len(traj_list) == 1 or not overlay:
        # Save one file per trajectory
        for idx, traj in enumerate(traj_list):
            plt.figure(figsize=(4, 4))
            plt.plot(traj.x, traj.y, "-o", markersize=2, linewidth=1)
            if traj.x.size > 0:
                plt.scatter([traj.x[0]], [traj.y[0]], color="green", s=20, label="start")
                plt.scatter([traj.x[-1]], [traj.y[-1]], color="red", s=20, label="end")
            plt.axis("equal")
            plt.title("Codling walk trajectory")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend(loc="best")
            plt.tight_layout()
            out_path = plot_dir / f"{fname_prefix}{theta_str}_{seed_str}_i{idx+1}.png"
            plt.savefig(out_path, dpi=150)
            if show:
                try:
                    plt.show()
                except Exception:
                    pass
            plt.close()
    else:
        # Overlay all trajectories in one figure with unique colors
        plt.figure(figsize=(5, 5))
        cmap = plt.get_cmap("tab20")
        for idx, traj in enumerate(traj_list):
            color = cmap(idx % cmap.N)
            plt.plot(traj.x, traj.y, linewidth=1.2, alpha=0.9, color=color, label=f"walker {idx+1}")
            if traj.x.size > 0:
                plt.scatter([traj.x[0]], [traj.y[0]], color=color, s=12, marker="o")
                plt.scatter([traj.x[-1]], [traj.y[-1]], color=color, s=12, marker="x")
        plt.axis("equal")
        plt.title(f"Population trajectories (N={len(traj_list)})")
        plt.xlabel("x")
        plt.ylabel("y")
        if len(traj_list) <= 15:
            plt.legend(loc="best", fontsize=8)
        else:
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6, ncol=1)
        plt.tight_layout()
        out_path = plot_dir / f"population_overlay{theta_str}_{seed_str}_N{len(traj_list)}.png"
        plt.savefig(out_path, dpi=150)
        if show:
            try:
                plt.show()
            except Exception:
                pass
        plt.close()


# Plot circular (rose) histograms of segment angles
def plot_segment_angle_distributions(
    trajectories: Trajectory | Sequence[Trajectory],
    *,
    overlay: bool = False,
    bins: int = 36,
    grid_cols: int = 4,
    out_dir: str | None = None,
    seed: int | None = None,
    show: bool = False,
    degrees: bool = False,
    fname_prefix: str = "angles_rose",
    theta: dict | None = None,
) -> None:
    """
    Plot circular (rose) histograms of segment angles for each walker.

    Parameters:
    - trajectories: Trajectory or sequence of Trajectory
    - overlay: if True, overlay all walkers in a single polar axis; otherwise,
               create a grid of polar subplots (single saved figure).
    - bins: number of angular bins (uniform over 2π).
    - grid_cols: number of columns in the grid layout for per-walker plots.
    - out_dir: directory to save figures; defaults to 'plot' next to this file.
    - seed: optional seed annotation for filenames.
    - show: whether to display figures.
    - degrees: if True, angle axis ticks use degrees; input data are always
               wrapped to [0, 2π) internally.
    - fname_prefix: filename prefix for saved figure.
    - theta: optional dict with keys 'kappa' and 'd_tau' for filename annotation.
    """
    try:
        from pathlib import Path

        import matplotlib.pyplot as plt
    except Exception:
        return

    # Normalize input
    if isinstance(trajectories, Trajectory):
        traj_list = [trajectories]
    else:
        traj_list = list(trajectories)

    # Prepare output
    plot_dir = Path(out_dir) if out_dir is not None else Path(__file__).parent / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)
    seed_str = f"seed_{int(seed)}" if seed is not None else "noseed"
    theta_str = ""
    if isinstance(theta, dict):
        try:
            kappa_val = float(theta.get("kappa"))
            dtau_val = float(theta.get("d_tau"))
            theta_str = f"_kappa{kappa_val:.3g}_dtau{dtau_val:.3g}"
        except Exception:
            pass

    # Helper: histogram on [0, 2π)
    def rose_hist(a: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
        if a.size == 0:
            # return empty
            edges = np.linspace(0.0, 2.0 * np.pi, bins + 1)
            centers = (edges[:-1] + edges[1:]) / 2.0
            return np.zeros_like(centers), centers
        ang = np.mod(a, 2.0 * np.pi)
        edges = np.linspace(0.0, 2.0 * np.pi, bins + 1)
        counts, _ = np.histogram(ang, bins=edges)
        prob = counts.astype(float)
        total = prob.sum()
        if total > 0:
            prob /= total  # probability mass per bin
        centers = (edges[:-1] + edges[1:]) / 2.0
        return prob, centers

    if overlay:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, 6))
        cmap = plt.get_cmap("tab20")
        for idx, traj in enumerate(traj_list):
            prob, centers = rose_hist(traj.segment_angles, bins)
            color = cmap(idx % cmap.N)
            width = 2.0 * np.pi / bins
            ax.bar(
                centers,
                prob,
                width=width,
                bottom=0.0,
                color=color,
                alpha=0.5,
                edgecolor="none",
                label=f"walker {idx+1}",
            )
        if degrees:
            ax.set_thetamin(0)
            ax.set_thetamax(360)
            ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
            ax.set_xticklabels([f"{d}°" for d in range(0, 360, 45)])
        ax.set_title(f"Segment angle distributions (N={len(traj_list)})")
        if len(traj_list) <= 15:
            ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=8)
        fig.tight_layout()
        out_path = plot_dir / f"{fname_prefix}_overlay{theta_str}_{seed_str}_N{len(traj_list)}.png"
        fig.savefig(out_path, dpi=150)
        if show:
            try:
                plt.show()
            except Exception:
                pass
        plt.close(fig)
    else:
        # Grid of polar subplots, single output file
        n = len(traj_list)
        cols = max(1, int(grid_cols))
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(
            rows, cols, subplot_kw=dict(polar=True), figsize=(4 * cols, 4 * rows)
        )
        # axes may be scalar if n==1
        axes_arr = (
            np.array(axes).reshape(rows, cols)
            if isinstance(axes, np.ndarray)
            else np.array([[axes]])
        )
        cmap = plt.get_cmap("tab20")
        for i, traj in enumerate(traj_list):
            r = i // cols
            c = i % cols
            ax = axes_arr[r, c]
            prob, centers = rose_hist(traj.segment_angles, bins)
            color = cmap(i % cmap.N)
            width = 2.0 * np.pi / bins
            ax.bar(centers, prob, width=width, bottom=0.0, color=color, alpha=0.8, edgecolor="none")
            ax.set_title(f"walker {i+1}")
            if degrees:
                ax.set_thetamin(0)
                ax.set_thetamax(360)
                ax.set_xticks(np.deg2rad(np.arange(0, 360, 90)))
                ax.set_xticklabels([f"{d}°" for d in range(0, 360, 90)])
        # Hide any unused axes
        for j in range(n, rows * cols):
            r = j // cols
            c = j % cols
            ax = axes_arr[r, c]
            ax.set_visible(False)
        fig.suptitle("Segment angle distributions", y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out_path = plot_dir / f"{fname_prefix}_grid{theta_str}_{seed_str}_N{len(traj_list)}.png"
        fig.savefig(out_path, dpi=150)
        if show:
            try:
                plt.show()
            except Exception:
                pass
        plt.close(fig)

# Plot chemo-attractant field
def plot_field(
    field_type: str = "linear",
    theta: dict | None = None,
    config: dict | None = None,
    xlim: tuple[float, float] = (-5, 5),
    ylim: tuple[float, float] = (-5, 5),
    resolution: int = 200,
    out_dir: str | None = None,
    show: bool = False,
    fname_prefix: str = "field",
) -> None:
    """
    Plot the chemo-attractant field as a 2D heatmap with contours.

    Parameters:
    - field_type: "linear" or "cancer"
    - theta: dict with field parameters (for cancer: target_Q, target_D, target_L)
    - xlim: x-axis range (min, max)
    - ylim: y-axis range (min, max)
    - resolution: number of grid points along each axis
    - out_dir: directory to save figure; defaults to 'plot' next to this file
    - show: whether to display the figure
    - fname_prefix: filename prefix for saved figure
    """
    try:
        from pathlib import Path

        import matplotlib.pyplot as plt
    except Exception:
        return

    # Prepare output
    plot_dir = Path(out_dir) if out_dir is not None else Path(__file__).parent / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Create field function
    if field_type == "cancer":
        if theta is None:
            raise ValueError("theta dict required for cancer field (keys: target_Q, target_D, target_L)")
        L = float(theta.get("target_L"))
        Q = float(config.get("target_Q"))
        x0 = float(config.get("target_x0"))
        y0 = float(config.get("target_y0"))
        C = lambda x, y: cancer_field(x, y, x0, y0, Q, L)
        title = f"Cancer Field (Q={Q:.1f}, λ={L:.2f})"
        fname = f"{fname_prefix}_cancer_Q{Q:.0f}_L{L:.2f}.png"
    else:
        C = default_field
        title = "Linear Field (C = 0.01·x)"
        fname = f"{fname_prefix}_linear.png"

    # Create grid
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    gradC_func = lambda x, y: central_gradient(C, x, y)

    # Evaluate field and gradient at each grid point
    U = np.zeros_like(X)  # gradient x-component
    V = np.zeros_like(Y)  # gradient y-component
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = C(X[i, j], Y[i, j])
            dCdx, dCdy = gradC_func(X[i, j], Y[i, j])
            U[i, j] = dCdx
            V[i, j] = dCdy

    # Plot field and gradient
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Left panel: Field concentration ---
    im1 = ax1.contourf(X, Y, Z, levels=50, cmap="viridis")
    contours1 = ax1.contour(X, Y, Z, levels=10, colors="white", alpha=0.4, linewidths=0.5)
    ax1.clabel(contours1, inline=True, fontsize=8, fmt="%.2e")
    cbar1 = fig.colorbar(im1, ax=ax1, label="Concentration C(x, y)")
    
    if field_type == "cancer":
        ax1.plot(x0, y0, "r*", markersize=15, label=f"Target ({x0}, {y0})")
    ax1.plot(0, 0, "ko", markersize=8, label="Origin (0, 0)")
    ax1.legend(loc="upper left")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(f"{title} - Concentration")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    
    # --- Right panel: Gradient magnitude with quiver ---
    grad_mag = np.sqrt(U**2 + V**2)
    im2 = ax2.contourf(X, Y, grad_mag, levels=50, cmap="plasma")
    cbar2 = fig.colorbar(im2, ax=ax2, label="Gradient Magnitude |∇C|")
    
    # Quiver plot (subsample for clarity)
    skip = max(1, resolution // 20)
    ax2.quiver(
        X[::skip, ::skip], 
        Y[::skip, ::skip], 
        U[::skip, ::skip], 
        V[::skip, ::skip],
        color="white",
        alpha=0.6,
        scale_units="xy",
        scale=None,
        width=0.003,
    )
    
    if field_type == "cancer":
        ax2.plot(x0, y0, "r*", markersize=15, label=f"Target ({x0}, {y0})")
    ax2.plot(0, 0, "ko", markersize=8, label="Origin (0, 0)")
    ax2.legend(loc="upper left")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(f"{title} - Gradient")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = plot_dir / fname
    plt.savefig(out_path, dpi=150)
    print(f"Field plot saved to: {out_path}")
    
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()
