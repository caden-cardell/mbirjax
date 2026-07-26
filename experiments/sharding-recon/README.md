# Reconstruction performance evaluation (sharding-recon)

Measures the wall-clock performance, memory footprint, and correctness of MBIRJAX cone-beam
`recon()` as a function of sinogram size and GPU count. Built to mirror the sibling
`denoiser.py` harness, but times the full `recon()` instead of `denoise()`.

## What gets measured

For each **sinogram shape** `(num_views, num_det_rows, num_det_channels)` and each **GPU count**
(1, 2, 4), one `recon()` call is timed (including one-time JIT compile — representative of a real
run). Every run appends a row with:

| group | columns |
|---|---|
| size | `num_views, num_det_rows, num_det_channels, sweep_axis, num_recon_voxels, num_sino_elements` |
| config | `max_iterations, stop_threshold_change_pct, num_iterations_run, seed` |
| devices | `cuda_visible_devices, num_gpus_visible, num_devices_used` |
| timing | `elapsed_seconds, sec_per_iteration` |
| memory | `peak_gpu_gb_total, peak_gpu_gb_per_device` (JSON), `peak_cpu_gb` |
| quality | `nrmse` (recon vs the cached clean phantom) |
| provenance | `mbirjax_version, git_commit, hostname, gpu_name, timestamp` |

## The sweep

Anisotropic: baseline **512³**, then increase exactly **one** sinogram dimension by 256 up to
2048 with the other two held at 512 (see `configs.sh`, 19 configs). The three axes are *not*
symmetric — `num_det_rows` drives the sharded recon-slice axis, so multi-GPU scaling is expected
to be strongest along the **rows** sweep. Sweeping one axis at a time isolates whether any single
dimension scales non-linearly.

## How to run (Gautschi cluster)

Paths in the sbatch scripts assume the repo at `/home/ncardel/repos/mbirjax` and a conda env
named `mbirjax`; adjust if yours differ. Cached data goes to `/scratch/gautschi/ncardel/recon_mem`
(set by `DATA_DIR` in `recon.py`).

### One command (recommended)

`run_all.sh` submits everything at once: the data-generation array plus the 1/2/4-GPU sweeps,
chained with a SLURM `aftercorr` dependency so each timing task starts as soon as *its own*
data is cached. Nothing blocks your shell.

```bash
cd experiments/sharding-recon
./run_all.sh                 # generate data + run 1, 2, 4-GPU sweeps
./run_all.sh --no-prep       # data already cached; just run the sweeps
./run_all.sh --gpus "1 4"    # only the listed GPU counts
squeue -u $USER              # monitor
python plot_results.py       # build figures + results/report.md when done
```

### Step by step (equivalent, if you prefer manual control)

```bash
cd experiments/sharding-recon

# Night 0 — generate & cache all phantoms+sinograms once (NOT timed). Do this before any sweep.
sbatch prepare_data.sh

# Nights 1-3 — one GPU count per night (each is a 19-task array).
sbatch recon_slurm_1gpu.sh
sbatch recon_slurm_2gpu.sh
sbatch recon_slurm_4gpu.sh

# Night 4+ — re-submit any sweep to accumulate repeats (error bars). Rows append to the
# per-(gpu, shape) CSVs under results/.

# Any time — regenerate figures + results/report.md from whatever CSVs exist so far.
python plot_results.py
```

Results layout:

```
results/
  recon_1gpu_512x512x512.csv        # one file per (gpu-count, shape); repeats append as rows
  recon_2gpu_512x2048x512.csv
  ...
  figures/fig_*.png                 # written by plot_results.py
  report.md                         # auto-generated report embedding the figures + tables
```

## Design notes / gotchas

- **GPU count is not a code path.** `recon.py` uses `use_gpu='automatic'`, which shards across
  *all visible* GPUs. The count is set purely by SLURM `--gpus-per-node` (→ `CUDA_VISIBLE_DEVICES`).
  `num_devices_used` in the CSV records what actually happened.
- **Timing includes JIT compile** (single-call methodology, matching `denoiser.py`). This inflates
  small-size points; interpret the smallest sizes with that in mind. It is deliberately the
  end-user-representative number.
- **Reproducibility.** `np.random.seed(SEED)` is set before every timed `recon()` so the stochastic
  pixel partitions — and hence NRMSE — are comparable across GPU counts and repeats.
- **One CSV per (gpu, shape)** avoids write races between concurrent array tasks and lets repeats
  simply append. `plot_results.py` globs them all.
- **OOM** at large sizes on few GPUs just aborts that array task (recorded as a missing point);
  the sweep is capped at 2048 by design.
- **No pandas needed** — `plot_results.py` uses only stdlib `csv` + numpy + matplotlib.

See `LAB_NOTEBOOK.md` for the per-night run log.
