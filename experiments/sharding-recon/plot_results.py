# -*- coding: utf-8 -*-
"""
Aggregate every results/*.csv produced by recon.py and emit the figure set + a markdown report.

Each recon.py run appends one row (one timed recon() call) to a per-(gpu-count, shape) CSV.
Re-submitting a sweep on a later night appends more rows, giving repeats. This script:

  1. loads all rows,
  2. groups them by (sweep axis, swept size, GPU count) and reduces repeats to median + IQR,
  3. writes figures (time / speedup / efficiency / memory / throughput / NRMSE vs size), and
  4. writes results/report.md embedding those figures and a summary + power-law-exponent table.

The baseline 512^3 point is shared: it is folded into all three axis series so each axis has a
common left anchor. No third-party data libraries are required -- stdlib csv + numpy + matplotlib.

Usage:
    python plot_results.py [results_dir]   # defaults to ./results next to this file
"""

import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AXES = ['views', 'rows', 'channels']
BASELINE = 512
# Which sinogram dimension each axis label varies (used to recover swept size from a row).
AXIS_DIM = {'views': 'num_views', 'rows': 'num_det_rows', 'channels': 'num_det_channels'}

FLOAT_COLS = ['elapsed_seconds', 'sec_per_iteration', 'peak_gpu_gb_total', 'peak_cpu_gb', 'nrmse']
INT_COLS = ['num_views', 'num_det_rows', 'num_det_channels', 'num_recon_voxels',
            'num_sino_elements', 'num_iterations_run', 'num_devices_used', 'num_gpus_visible']


def load_rows(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, '*.csv'))):
        with open(path, newline='') as f:
            for raw in csv.DictReader(f):
                row = dict(raw)
                for c in FLOAT_COLS:
                    try:
                        row[c] = float(raw.get(c, ''))
                    except (TypeError, ValueError):
                        row[c] = float('nan')
                for c in INT_COLS:
                    try:
                        row[c] = int(float(raw.get(c, '')))
                    except (TypeError, ValueError):
                        row[c] = None
                rows.append(row)
    return rows


def swept_size(row):
    """The size of the one axis being swept (the other two sit at BASELINE); BASELINE for baseline."""
    dims = {'views': row['num_views'], 'rows': row['num_det_rows'], 'channels': row['num_det_channels']}
    changed = [d for d, v in dims.items() if v != BASELINE]
    if not changed:
        return BASELINE
    return dims[changed[0]]


def axis_series(rows, axis):
    """Rows that belong to `axis`: its own swept rows plus the shared baseline point."""
    out = []
    for row in rows:
        sa = row.get('sweep_axis')
        if sa == axis or sa == 'baseline':
            out.append(row)
    return out


def aggregate(rows, value_col):
    """Return {gpu_count: (sizes, median, q25, q75, n)} for value_col over an axis' rows."""
    by_gpu = {}
    for row in rows:
        n = row.get('num_devices_used')
        if not n:
            continue
        by_gpu.setdefault(n, {}).setdefault(swept_size(row), []).append(row[value_col])
    result = {}
    for gpu, size_map in by_gpu.items():
        sizes = sorted(size_map)
        med = np.array([np.nanmedian(size_map[s]) for s in sizes])
        q25 = np.array([np.nanpercentile(size_map[s], 25) for s in sizes])
        q75 = np.array([np.nanpercentile(size_map[s], 75) for s in sizes])
        cnt = np.array([len(size_map[s]) for s in sizes])
        result[gpu] = (np.array(sizes, dtype=float), med, q25, q75, cnt)
    return result


def power_law_exponent(sizes, values):
    """Slope of log(value) vs log(size): value ~ size^k. NaN if too few finite points."""
    mask = np.isfinite(sizes) & np.isfinite(values) & (sizes > 0) & (values > 0)
    if mask.sum() < 2:
        return float('nan')
    k, _ = np.polyfit(np.log(sizes[mask]), np.log(values[mask]), 1)
    return float(k)


def _gpu_style(gpu):
    colors = {1: 'C0', 2: 'C1', 4: 'C2'}
    return colors.get(gpu, 'C3')


# ----------------------------------------------------------------------------- figures

def fig_metric_vs_size(rows_by_axis, value_col, ylabel, title, out_path, logy=True,
                       annotate_exponent=False):
    fig, axgrid = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, axis_name in zip(axgrid, AXES):
        agg = aggregate(rows_by_axis[axis_name], value_col)
        for gpu in sorted(agg):
            sizes, med, q25, q75, _ = agg[gpu]
            label = '{} GPU'.format(gpu)
            if annotate_exponent:
                k = power_law_exponent(sizes, med)
                label += '  (k={:.2f})'.format(k)
            ax.plot(sizes, med, marker='o', color=_gpu_style(gpu), label=label)
            ax.fill_between(sizes, q25, q75, color=_gpu_style(gpu), alpha=0.15)
        ax.set_title('{} sweep'.format(axis_name))
        ax.set_xlabel('swept dimension size')
        ax.set_xscale('log', base=2)
        if logy:
            ax.set_yscale('log', base=2)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)
    axgrid[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def fig_scaling(rows_by_axis, out_path, efficiency=False):
    """Speedup S(N)=T(1)/T(N) (or efficiency S(N)/N) vs size, per axis."""
    fig, axgrid = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, axis_name in zip(axgrid, AXES):
        agg = aggregate(rows_by_axis[axis_name], 'elapsed_seconds')
        if 1 not in agg:
            ax.set_title('{} sweep (no 1-GPU baseline)'.format(axis_name))
            continue
        base_sizes, base_med, _, _, _ = agg[1]
        base = dict(zip(base_sizes, base_med))
        for gpu in sorted(agg):
            if gpu == 1:
                continue
            sizes, med, _, _, _ = agg[gpu]
            xs, ys = [], []
            for s, t in zip(sizes, med):
                if s in base and np.isfinite(base[s]) and np.isfinite(t) and t > 0:
                    speedup = base[s] / t
                    xs.append(s)
                    ys.append(speedup / gpu if efficiency else speedup)
            ax.plot(xs, ys, marker='o', color=_gpu_style(gpu), label='{} GPU'.format(gpu))
        # ideal reference
        xlim = sorted({s for s in base_sizes})
        if efficiency:
            ax.axhline(1.0, ls='--', color='k', alpha=0.5, label='ideal')
        else:
            for gpu in sorted(agg):
                if gpu != 1:
                    ax.axhline(gpu, ls='--', color=_gpu_style(gpu), alpha=0.4)
        ax.set_title('{} sweep'.format(axis_name))
        ax.set_xlabel('swept dimension size')
        ax.set_xscale('log', base=2)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)
    axgrid[0].set_ylabel('parallel efficiency S(N)/N' if efficiency else 'speedup T(1)/T(N)')
    fig.suptitle('Parallel efficiency vs problem size' if efficiency
                 else 'Strong-scaling speedup vs problem size')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def fig_throughput(rows_by_axis, out_path):
    """Throughput = num_recon_voxels * num_iterations_run / elapsed, vs size, per axis."""
    # Build a derived per-row throughput first, then aggregate.
    for rows in rows_by_axis.values():
        for row in rows:
            it = row.get('num_iterations_run') or 0
            vox = row.get('num_recon_voxels') or 0
            t = row.get('elapsed_seconds')
            row['throughput'] = (vox * it / t) if (t and t > 0) else float('nan')
    fig_metric_vs_size(rows_by_axis, 'throughput', 'voxel-iterations / second',
                       'Throughput vs problem size', out_path, logy=True)


# ----------------------------------------------------------------------------- report

def write_report(rows, rows_by_axis, results_dir, figure_files):
    lines = []
    lines.append('# Reconstruction performance report\n')
    lines.append('Auto-generated by `plot_results.py`. Each point is the **median** over repeats; '
                 'shaded bands / ± are the interquartile range (25–75%).\n')

    n_runs = len(rows)
    gpu_counts = sorted({r['num_devices_used'] for r in rows if r.get('num_devices_used')})
    commits = sorted({r.get('git_commit', '') for r in rows if r.get('git_commit')})
    versions = sorted({r.get('mbirjax_version', '') for r in rows if r.get('mbirjax_version')})
    gpus = sorted({r.get('gpu_name', '') for r in rows if r.get('gpu_name')})
    lines.append('- Total runs: **{}**'.format(n_runs))
    lines.append('- GPU counts: {}'.format(', '.join(str(g) for g in gpu_counts) or 'n/a'))
    lines.append('- mbirjax version(s): {}'.format(', '.join(versions) or 'n/a'))
    lines.append('- git commit(s): {}'.format(', '.join(commits) or 'n/a'))
    lines.append('- GPU model(s): {}\n'.format(', '.join(gpus) or 'n/a'))

    # Power-law exponents table (time vs size).
    lines.append('## Time scaling exponents (elapsed ∝ size^k)\n')
    lines.append('Fitted on the median timings. k≈1 is linear in the swept dimension; '
                 'k>1 flags a super-linear axis.\n')
    header = '| axis | ' + ' | '.join('{} GPU'.format(g) for g in gpu_counts) + ' |'
    lines.append(header)
    lines.append('|' + '---|' * (len(gpu_counts) + 1))
    for axis_name in AXES:
        agg = aggregate(rows_by_axis[axis_name], 'elapsed_seconds')
        cells = []
        for g in gpu_counts:
            if g in agg:
                sizes, med, _, _, _ = agg[g]
                cells.append('{:.2f}'.format(power_law_exponent(sizes, med)))
            else:
                cells.append('—')
        lines.append('| {} | {} |'.format(axis_name, ' | '.join(cells)))
    lines.append('')

    # Figures.
    lines.append('## Figures\n')
    captions = {
        'time': 'Elapsed recon time vs swept dimension (log–log). Legend shows the fitted '
                'power-law exponent k per GPU count.',
        'speedup': 'Strong-scaling speedup T(1)/T(N). Dashed lines mark the ideal N×.',
        'efficiency': 'Parallel efficiency S(N)/N. 1.0 (dashed) is perfect scaling.',
        'memory': 'Peak GPU memory (total across devices) vs swept dimension.',
        'throughput': 'Throughput (voxel-iterations per second) vs swept dimension.',
        'nrmse': 'Reconstruction NRMSE vs phantom — a correctness sanity check; should stay low '
                 'and roughly flat across GPU counts.',
    }
    for key, path in figure_files.items():
        lines.append('### {}\n'.format(key.capitalize()))
        lines.append('![{}]({})\n'.format(key, os.path.basename(path)))
        lines.append('{}\n'.format(captions.get(key, '')))

    # Full summary table.
    lines.append('## Summary table (median over repeats)\n')
    lines.append('| axis | size | GPUs | n | elapsed (s) | s/iter | peak GPU (GB) | '
                 'peak CPU (GB) | NRMSE |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for axis_name in AXES:
        arows = rows_by_axis[axis_name]
        el = aggregate(arows, 'elapsed_seconds')
        spi = aggregate(arows, 'sec_per_iteration')
        gmem = aggregate(arows, 'peak_gpu_gb_total')
        cmem = aggregate(arows, 'peak_cpu_gb')
        nr = aggregate(arows, 'nrmse')
        for gpu in sorted(el):
            sizes, med, _, _, cnt = el[gpu]
            spi_map = dict(zip(spi[gpu][0], spi[gpu][1])) if gpu in spi else {}
            gmem_map = dict(zip(gmem[gpu][0], gmem[gpu][1])) if gpu in gmem else {}
            cmem_map = dict(zip(cmem[gpu][0], cmem[gpu][1])) if gpu in cmem else {}
            nr_map = dict(zip(nr[gpu][0], nr[gpu][1])) if gpu in nr else {}
            for s, t, c in zip(sizes, med, cnt):
                lines.append('| {} | {:.0f} | {} | {} | {:.2f} | {:.3f} | {:.2f} | {:.2f} | {:.4f} |'
                             .format(axis_name, s, gpu, c, t, spi_map.get(s, float('nan')),
                                     gmem_map.get(s, float('nan')), cmem_map.get(s, float('nan')),
                                     nr_map.get(s, float('nan'))))
    lines.append('')

    report_path = os.path.join(results_dir, 'report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    return report_path


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, 'results')
    rows = load_rows(results_dir)
    if not rows:
        print('No result CSVs found in {} -- nothing to plot.'.format(results_dir))
        return

    # Pre-split rows per axis (each axis includes the shared baseline point).
    rows_by_axis = {axis: axis_series(rows, axis) for axis in AXES}

    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    figure_files = {
        'time': os.path.join(figures_dir, 'fig_time_vs_size.png'),
        'speedup': os.path.join(figures_dir, 'fig_speedup.png'),
        'efficiency': os.path.join(figures_dir, 'fig_efficiency.png'),
        'memory': os.path.join(figures_dir, 'fig_peak_gpu_memory.png'),
        'throughput': os.path.join(figures_dir, 'fig_throughput.png'),
        'nrmse': os.path.join(figures_dir, 'fig_nrmse.png'),
    }

    fig_metric_vs_size(rows_by_axis, 'elapsed_seconds', 'elapsed recon time (s)',
                       'Recon time vs problem size', figure_files['time'],
                       logy=True, annotate_exponent=True)
    fig_scaling(rows_by_axis, figure_files['speedup'], efficiency=False)
    fig_scaling(rows_by_axis, figure_files['efficiency'], efficiency=True)
    fig_metric_vs_size(rows_by_axis, 'peak_gpu_gb_total', 'peak GPU memory (GB)',
                       'Peak GPU memory vs problem size', figure_files['memory'], logy=False)
    fig_throughput(rows_by_axis, figure_files['throughput'])
    fig_metric_vs_size(rows_by_axis, 'nrmse', 'NRMSE',
                       'Reconstruction NRMSE vs problem size', figure_files['nrmse'], logy=False)

    report_path = write_report(rows, rows_by_axis, results_dir, figure_files)
    print('Wrote {} figures to {}'.format(len(figure_files), figures_dir))
    print('Wrote report to {}'.format(report_path))


if __name__ == '__main__':
    main()
