import csv
from pathlib import Path
import matplotlib.pyplot as plt


def read_csv_columns(csv_path: Path, x_idx: int, y_idx: int):
    xs = []
    ys = []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        # Peek header to decide if first row is non-numeric
        first_row = next(reader, None)
        if first_row is None:
            return xs, ys, None
        # Try to parse as float; if fails, treat as header
        def is_float(s: str) -> bool:
            try:
                float(s)
                return True
            except Exception:
                return False
        has_header = not (is_float(first_row[x_idx]) and is_float(first_row[y_idx]))
        if not has_header:
            xs.append(float(first_row[x_idx]))
            ys.append(float(first_row[y_idx]))
        header = first_row if has_header else None
        for row in reader:
            # Skip rows shorter than required indices
            if max(x_idx, y_idx) >= len(row):
                continue
            try:
                xs.append(float(row[x_idx]))
                ys.append(float(row[y_idx]))
            except ValueError:
                # Skip non-numeric rows
                continue
    return xs, ys, header


def main():
    # Column indices: plot second (1) vs third (2)
    x_idx = 0
    y_idx = 3
    # Hardcoded path and columns: first (0) vs third (2)
    csv_path = (Path(__file__).resolve().parent / "sbi-logs" / "SNPE" / "test.csv").resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    xs, ys, header = read_csv_columns(csv_path, x_idx, y_idx)
    if not xs or not ys:
        raise ValueError("No numeric data found for selected columns (0 and 2)")

    xlabel = header[x_idx] if header else f"col_{x_idx}"
    ylabel = header[y_idx] if header else f"col_{y_idx}"

    plt.figure(figsize=(7, 4))
    plt.scatter(xs, ys, s=12, alpha=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{xlabel} vs {ylabel}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
