#!/usr/bin/env python3
"""Render a recon-results table for psi_data_2 as a PNG suitable for slides."""

from pathlib import Path
import matplotlib.pyplot as plt

ROWS = [
    (2, "128³",  "41.16 s"),
    (2, "256³",  "44.78 s"),
    (2, "512³",  "107.96 s"),
    (2, "1024³", "792.38 s"),
    (2, "1280³", "2039.44 s"),
    (2, "1536³", "4383.84 s"),
    (2, "1792³", "FAILED — pre-flight memory check: insufficient GPU memory for sharded recon"),
    (2, "2048³", "FAILED — pre-flight memory check: insufficient GPU memory for sharded recon"),
    (4, "128³",  "42.91 s"),
    (4, "256³",  "45.70 s"),
    (4, "512³",  "85.49 s"),
    (4, "1024³", "523.70 s"),
    (4, "1280³", "1138.09 s"),
    (4, "1536³", "2237.61 s"),
    (4, "1792³", "FAILED — OOM during fdk_filter (21.44 GiB allocation)"),
    (4, "2048³", "FAILED — OOM during fdk_filter (32.00 GiB allocation)"),
]

HEADERS = ["GPUs", "Sinogram size", "Recon time / failure reason"]
OUTPUT = Path(__file__).parent / "psi_data_2_table.png"


def main():
    # Collapse the GPU column: only show the number on the first row of each group.
    display_rows = []
    prev_gpu = None
    for gpu, size, result in ROWS:
        gpu_cell = "" if gpu == prev_gpu else str(gpu)
        display_rows.append([gpu_cell, size, result])
        prev_gpu = gpu

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.axis("off")
    ax.set_title("PSI dataset — reconstruction time by GPU count and sinogram size",
                 fontsize=15, pad=14, weight="bold")

    table = ax.table(
        cellText=display_rows,
        colLabels=HEADERS,
        colWidths=[0.10, 0.18, 0.72],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    header_color = "#1f77b4"
    fail_color = "#fde2e2"
    group_colors = {2: "#f5f9ff", 4: "#fffaf0"}

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
            continue

        gpu, _, result = ROWS[row - 1]
        if "FAILED" in result:
            cell.set_facecolor(fail_color)
            if col == 2:
                cell.set_text_props(color="#a40000", weight="bold")
        else:
            cell.set_facecolor(group_colors[gpu])

        if col == 0 and display_rows[row - 1][0]:
            cell.set_text_props(weight="bold")

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
