#!/usr/bin/env python3
"""
Generate a test dataset CSV containing rows of:
    kappa, d_tau, angle_1, angle_2, ..., angle_M

Notes:
- For each sampled (kappa, d_tau), we simulate ONE trajectory and
    interpolate to integer times, then take its segment_angles.
- The number of angle columns equals floor(T_max).
- Output CSV has a header with column names.

Run:
    python code/generate_dataset.py --n 100 --tmax 100 --seed 123 --out data/test_angles.csv
"""
import argparse
from pathlib import Path
import numpy as np

from simulator_trajectory import simulate_codling_walk


def generate_rows(n: int, T_max: float, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)

    # Define sampling ranges for theta
    kappa_min, kappa_max = 1.0, 8.0
    dtau_min, dtau_max = 0.01, 0.15

    rows = []
    M = int(np.floor(T_max)) # number of segment angles after integer interpolation

    for _ in range(n):
        kappa = float(rng.uniform(kappa_min, kappa_max))
        d_tau = float(rng.uniform(dtau_min, dtau_max))

        theta = {"kappa": kappa, "d_tau": d_tau}
        config = {
            "T_max": float(T_max),
            "lambda": 0.5,
            "s": 1.0,
            "seed": int(rng.integers(1_000_000_000)),
        }
        traj = simulate_codling_walk(theta=theta, config=config)
        interpolated = traj.interpolate_to_integers(T_max)
        # Ensure fixed length M by padding/truncating
        angles = interpolated.segment_angles
        if angles.size < M:
            pad = np.full(M, angles[-1] if angles.size > 0 else 0.0, dtype=np.float32)
            pad[: angles.size] = angles
            angles_fixed = pad
        else:
            angles_fixed = angles[:M].astype(np.float32)
        # Wrap angles to [-pi, pi)
        angles_fixed = ((angles_fixed + np.pi) % (2.0 * np.pi)) - np.pi

        row = np.concatenate([[kappa, d_tau], angles_fixed.astype(np.float64)])
        rows.append(row)

    return np.vstack(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate test dataset of segment angles")
    parser.add_argument("--n", type=int, default=100, help="Number of samples (rows)")
    parser.add_argument("--tmax", type=float, default=100.0, help="Total simulation horizon T_max")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--out", type=str, default="data/test_angles.csv", help="Output CSV path")
    args = parser.parse_args()

    rows = generate_rows(n=args.n, T_max=args.tmax, seed=args.seed)

    # Ensure output directory exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build header aligned with generated row length
    M = int(np.floor(args.tmax))
    header_cols = ["kappa", "d_tau"] + [f"angle_{i+1}" for i in range(M)]
    header = ",".join(header_cols)

    # Sanity check: columns match header
    assert rows.shape[1] == 2 + M, (
        f"Row width ({rows.shape[1]}) != header width ({2+M}); "
        f"check T_max/angle generation."
    )

    np.savetxt(out_path, rows, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"Wrote {rows.shape[0]} rows to {out_path.resolve()} with {M} angle columns.")


if __name__ == "__main__":
    main()
