from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

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
    theta: dict,  # {"kappa": float, "d_tau": float}
    config: dict,  # {"s","lambda","T_max","B","seed","C","gradC","theta_init"}
    rng: np.random.Generator | None = None,
) -> Trajectory:
    """
    theta keys:
    - kappa (float): concentration parameter of von Mises turn angle distribution
    - d_tau (float): chemo sensitivity parameter for biasing turn angles
    config keys:
    - T_max (float): total simulation horizon (time units). Default: 100.0
    - lambda (float): Poisson turn rate λ (events per unit time). Default: 0.5
    - s (float): constant speed (distance units per time unit). Default: 1.0
    - C (Field|None): chemo-attractant field C(x, y). Default: linear field 0.01·x
    - gradC (Gradient|None): gradient ∇C(x, y); if None, uses central differences. Default: None
    - theta_init (float|None): initial heading angle in radians; if None, draws U[0, 2π). Default: None
    - seed (int|None): RNG seed for reproducible trajectory. Default: None
    """
    # extract parameters from dicts
    kappa = float(theta.get("kappa"))
    d_tau = float(theta.get("d_tau"))

    s = float(config.get("s", 1.0))
    lambda_ = float(config.get("lambda", 0.5))
    T_max = float(config.get("T_max", 100.0))
    C = config.get("C")
    gradC = config.get("gradC")
    theta_init = config.get("theta_init")
    # initialize local RNG (avoid global state)
    if rng is None:
        seed = config.get("seed")
        try:
            rng = np.random.default_rng(None if seed is None else int(seed))
        except Exception:
            rng = np.random.default_rng()

    if C is None:
        C = default_field
    if gradC is None:
        gradC = lambda x, y, h=1e-3: central_gradient(C, x, y, h)

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

        dCdx, dCdy = gradC(x, y)
        if dCdx == 0.0 and dCdy == 0.0:
            theta_pref = theta
        else:
            theta_pref = np.arctan2(dCdy, dCdx)

        mu = -d_tau * (theta - theta_pref)
        delta_theta = rng.vonmises(mu, kappa)
        theta += delta_theta

        x_new = x + s * dt * np.cos(theta)
        y_new = y + s * dt * np.sin(theta)
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
    theta: dict,  # {"kappa": float, "d_tau": float}
    config: dict,  # {"N_pop","s","lambda","T_max","B","seed", ...}
    rng: np.random.Generator | None = None,
) -> Sequence[Trajectory]:
    """
    theta keys:
    - kappa (float): concentration parameter of von Mises turn angle distribution
    - d_tau (float): chemo sensitivity parameter for biasing turn angles
    config keys:
    - N_pop (int): number of walkers. Default: 1
    - T_max (float): total simulation horizon (time units). Default: 100.0
    - lambda (float): Poisson turn rate λ (events per unit time). Default: 0.5
    - s (float): constant speed (distance units per time unit). Default: 1.0
    - C (Field|None): chemo-attractant field C(x, y). Default: linear field 0.01·x
    - gradC (Gradient|None): gradient ∇C(x, y); if None, uses central differences. Default: None
    - theta_init (float|None): initial heading angle in radians; if None, draws U[0, 2π). Default: None
    - seed (int|None): RNG seed for reproducible trajectories. Default: None
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
    theta: dict,  # {"kappa", "d_tau"}
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
