#!/usr/bin/env python3
"""
Plot reconstruction time vs sinogram size for 1, 2, and 4 GPU configurations.

Usage:
    python plot_recon_times.py [--dir DATA_DIR] [--output OUTPUT_FILE]

If --dir is not given, the script's own directory is used.
If --output is not given, the plot is displayed interactively.
"""

import argparse
import re
import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Map GPU count to plot style
GPU_STYLES = {
    1: {"color": "#1f77b4", "marker": "o", "label": "1 GPU"},
    2: {"color": "#ff7f0e", "marker": "s", "label": "2 GPUs"},
    4: {"color": "#2ca02c", "marker": "^", "label": "4 GPUs"},
}

FILENAME_RE = re.compile(r"recon_time_(\d+)x(\d+)x(\d+)_(\d+)gpu\.txt$")


def parse_file(path: Path) -> dict | None:
    """Return parsed fields from a result file, or None on failure."""
    m = FILENAME_RE.search(path.name)
    if not m:
        print(f"Skipping {path.name}: filename does not match expected pattern.")
        return None

    num_gpus = int(m.group(4))

    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"Skipping {path.name}: no data rows.")
        return None

    row = rows[0]
    return {
        "num_views": int(row["num_views"]),
        "num_det_rows": int(row["num_det_rows"]),
        "num_det_channels": int(row["num_det_channels"]),
        "elapsed_seconds": float(row["elapsed_seconds"]),
        "num_gpus": num_gpus,
    }


def main():
    parser = argparse.ArgumentParser(description="Plot MBIRJAX reconstruction times.")
    parser.add_argument("--dir", default=None, help="Directory containing recon_time_*.txt files (default: script directory)")
    parser.add_argument("--output", default=None, help="Save plot to this file instead of displaying interactively")
    parser.add_argument("--axis", choices=["linear", "loglog"], default="linear", help="Axis scale (default: linear)")
    args = parser.parse_args()

    search_dir = Path(args.dir) if args.dir else Path(__file__).parent
    txt_files = sorted(search_dir.glob("recon_time_*.txt"))

    if not txt_files:
        print(f"No recon_time_*.txt files found in {search_dir}.")
        sys.exit(1)

    # Group results by GPU count: {num_gpus: [(size, elapsed_seconds), ...]}
    data: dict[int, list[tuple[int, float]]] = {}

    for path in txt_files:
        result = parse_file(path)
        if result is None:
            continue

        # Use num_views as the representative sinogram size (all three dims are equal)
        size = result["num_views"]
        gpu_count = result["num_gpus"]

        data.setdefault(gpu_count, []).append((size, result["elapsed_seconds"]))

    if not data:
        print("No valid data files found.")
        sys.exit(1)

    # Sort each series by size
    for gpu_count in data:
        data[gpu_count].sort(key=lambda t: t[0])

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))

    for gpu_count in sorted(data.keys()):
        style = GPU_STYLES.get(gpu_count, {"color": "gray", "marker": "x", "label": f"{gpu_count} GPUs"})
        sizes, times = zip(*data[gpu_count])
        ax.plot(
            sizes,
            times,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Sinogram size (N, for N×N×N)", fontsize=13)
    ax.set_ylabel("Reconstruction time (seconds)", fontsize=13)
    ax.set_title("MBIRJAX Reconstruction Time vs Sinogram Size", fontsize=14)
    ax.legend(fontsize=12)
    if args.axis == "loglog":
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(sorted({size for sizes_times in data.values() for size, _ in sizes_times}))
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis="both", labelsize=11)

    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Plot saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
