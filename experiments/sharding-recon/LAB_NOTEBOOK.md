# Lab notebook — recon performance evaluation

Chronological log of every batch submitted. One entry per night. Keep it factual: what was
submitted, the SLURM job IDs, what finished, what failed (OOM / errors), and any observation
worth revisiting. This is the documented process the assignment asks for.

Template to copy for each entry:

---

## YYYY-MM-DD — <one-line summary>

- **mbirjax commit:** `<git short sha>`  |  **env:** `mbirjax`  |  **node/GPU:** `<gpu_name>`
- **Submitted:**
  - `sbatch <script>` → job `<jobid>` (array 0–18, <N> GPU)
- **Completed:** <which array tasks / shapes finished>
- **Failed / OOM:** <shape(s) + reason; check logs/*.err>
- **Observations:** <e.g. rows sweep scales near-linearly with GPUs; views sweep flat; anomaly at X>
- **Report regenerated:** `python plot_results.py` → yes/no

---

## Planned schedule

| night | action |
|---|---|
| 0 | `sbatch prepare_data.sh` — cache all 19 phantoms+sinograms (once) |
| 1 | `sbatch recon_slurm_1gpu.sh` — single-GPU reference + per-axis exponents |
| 2 | `sbatch recon_slurm_2gpu.sh` |
| 3 | `sbatch recon_slurm_4gpu.sh` |
| 4 | re-submit all three sweeps for repeats (error bars); re-run anomalies |
| 5+ | buffer: finer steps on the most interesting axis; regenerate report |

## Running log

<!-- Add newest entries at the top. -->
