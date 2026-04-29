#!/usr/bin/env python3
"""
Plot reconstruction time vs sinogram size for 1, 2, and 4 GPU configurations.

Usage:
    python plot_recon_times.py [--dir DATA_DIR [LABEL]] [--output OUTPUT_FILE] [--axis loglog]

--dir can be specified multiple times to overlay datasets on the same plot.
An optional label after the directory path is used in the legend; if omitted
the directory name is used.

Examples:
    python plot_recon_times.py --dir ./psi_data PSI --dir ./usi_data USI
    python plot_recon_times.py --dir ./results --axis loglog --output plot.png
"""

import argparse
import re
import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Markers and line styles cycled across datasets when multiple --dir are given
DATASET_STYLES = [
    {"linestyle": "-",  "color_set": ["#1f77b4", "#ff7f0e", "#2ca02c"]},
    {"linestyle": "--", "color_set": ["#d62728", "#9467bd", "#8c564b"]},
    {"linestyle": ":",  "color_set": ["#e377c2", "#7f7f7f", "#bcbd22"]},
]

GPU_MARKERS = {1: "o", 2: "s", 4: "^"}

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
        "elapsed_seconds": float(row["elapsed_seconds"]),
        "num_gpus": num_gpus,
    }


def load_dir(search_dir: Path) -> dict[int, list[tuple[int, float]]]:
    """Return {num_gpus: [(size, elapsed_seconds), ...]} for a directory."""
    txt_files = sorted(search_dir.glob("recon_time_*.txt"))
    if not txt_files:
        print(f"No recon_time_*.txt files found in {search_dir}.")
        return {}

    data: dict[int, list[tuple[int, float]]] = {}
    for path in txt_files:
        result = parse_file(path)
        if result is None:
            continue
        size = result["num_views"]
        gpu_count = result["num_gpus"]
        data.setdefault(gpu_count, []).append((size, result["elapsed_seconds"]))

    for gpu_count in data:
        data[gpu_count].sort(key=lambda t: t[0])

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Plot MBIRJAX reconstruction times.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        nargs="+",
        action="append",
        metavar=("DATA_DIR", "LABEL"),
        help="Directory of recon_time_*.txt files, with an optional legend label. "
             "May be repeated to overlay multiple datasets.",
    )
    parser.add_argument("--output", default=None, help="Save plot to this file instead of displaying interactively")
    parser.add_argument("--axis", choices=["linear", "loglog"], default="linear", help="Axis scale (default: linear)")
    parser.add_argument("--n3-scale", type=float, default=None,
                        help="Scale factor for the N^3 reference line (y = scale * N^3). "
                             "Defaults to the best observed time/N^3 ratio in the data.")
    args = parser.parse_args()

    # Build list of (Path, label) pairs
    if args.dir:
        dirs = []
        for entry in args.dir:
            path = Path(entry[0])
            label = entry[1] if len(entry) > 1 else path.name
            dirs.append((path, label))
    else:
        default_dir = Path(__file__).parent
        dirs = [(default_dir, default_dir.name)]

    # Load data for each directory
    datasets = []
    for search_dir, label in dirs:
        data = load_dir(search_dir)
        if data:
            datasets.append((label, data))

    if not datasets:
        print("No valid data files found.")
        sys.exit(1)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))

    all_sizes = set()
    all_points: list[tuple[int, float]] = []  # collect (size, time) for N^3 scaling
    for ds_idx, (label, data) in enumerate(datasets):
        style = DATASET_STYLES[ds_idx % len(DATASET_STYLES)]
        for gpu_idx, gpu_count in enumerate(sorted(data.keys())):
            color = style["color_set"][gpu_idx % len(style["color_set"])]
            marker = GPU_MARKERS.get(gpu_count, "x")
            gpu_label = f"{gpu_count} GPU" if gpu_count == 1 else f"{gpu_count} GPUs"
            legend_label = f"{label} – {gpu_label}" if len(datasets) > 1 else gpu_label
            sizes, times = zip(*data[gpu_count])
            all_sizes.update(sizes)
            all_points.extend(zip(sizes, times))
            ax.plot(
                sizes,
                times,
                color=color,
                marker=marker,
                linestyle=style["linestyle"],
                label=legend_label,
                linewidth=2,
                markersize=8,
            )
            # Annotate the largest-size point for this (dataset, gpu_count) with its elapsed time.
            max_size, max_time = sizes[-1], times[-1]
            ax.annotate(
                f"{max_time:.1f}s",
                xy=(max_size, max_time),
                xytext=(-6, 8),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=10,
                color=color,
            )

    # N^3 reference line
    sorted_sizes = sorted(all_sizes)
    if args.n3_scale is not None:
        n3_scale = args.n3_scale
    else:
        n3_scale = min(t / (n ** 3) for n, t in all_points)
    n3_times = [n3_scale * n ** 3 for n in sorted_sizes]
    ax.plot(
        sorted_sizes,
        n3_times,
        color="black",
        marker="*",
        linestyle="--",
        label="O(N³) reference",
        linewidth=1.5,
        markersize=10,
        zorder=0,
    )

    ax.set_xlabel("Sinogram size (N, for N×N×N)", fontsize=13)
    ax.set_ylabel("Reconstruction time (seconds)", fontsize=13)
    ax.set_title("MBIRJAX Reconstruction Time vs Sinogram Size", fontsize=14)
    ax.legend(fontsize=11)

    if args.axis == "loglog":
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(sorted(all_sizes))
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
