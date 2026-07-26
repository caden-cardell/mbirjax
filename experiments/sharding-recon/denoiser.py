# -*- coding: utf-8 -*-
"""
Time only the denoise() call, using a phantom cached on disk.

Generating a large Shepp-Logan phantom is itself non-trivial work, and it would otherwise
be repeated (and add noise to the timing) on every run of a scaling/GPU-count sweep. This
script caches the clean phantom for a given size as an HDF5 file the first time it is
asked for, and loads it from disk on every later call -- so phantom generation/IO happens
once per size and is excluded from the timed region. Only denoiser.denoise() is timed.

Noise is added fresh each run (cheap, in memory) from SEED, so the same cached phantom
can be reused across different sigma-noise-added values while staying reproducible.

Edit the constants below to change what a given run does. SIZE can also be overridden from
the command line: `python denoiser_timing_cached_phantom.py 512`, and OUTPUT_CSV with a second
argument: `python denoiser_timing_cached_phantom.py 512 denoiser_scaling_results_v0.6.17.1.csv`
(useful for keeping results from different mbirjax versions in separate files).
"""

import csv
import json
import os
import sys
import time

import numpy as np
import jax
import mbirjax as mj

# ##########################
# Edit these to change the run
SIZE = 256  # Cube edge length; sets num_views = num_det_rows = num_det_channels.
PHANTOM_DIR = '/scratch/gautschi/ncardel/qggmrf-denoiser'  # Holds cached phantom_size{N}.h5 files.
SIGMA_NOISE_ADDED = 0.1  # Std of the Gaussian noise added to the (cached) phantom.
SIGMA_NOISE = None  # Noise std passed to denoise(). None means estimate it from the image.
SHARPNESS = 0.0
STOP_THRESHOLD_CHANGE_PCT = 0.0
MAX_ITERATIONS = 15
SEED = 42  # Seed for the phantom noise, so repeated sizes are comparable.
OUTPUT_CSV = 'denoiser_scaling_results.csv'  # CSV file to append this run's results to.
# ##########################


def get_or_create_phantom(size, phantom_dir):
    """Load the cached phantom for this size, generating and caching it first if needed."""
    recon_shape = (size, size, size)
    phantom_path = os.path.join(phantom_dir, 'phantom_size{}.h5'.format(size))

    if os.path.isfile(phantom_path):
        print('Loading cached phantom from {}'.format(phantom_path))
        phantom, _ = mj.load_data_hdf5(phantom_path)
    else:
        print('No cached phantom found; generating and saving to {}'.format(phantom_path))
        phantom = mj.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
        mj.save_data_hdf5(phantom_path, phantom, array_name='phantom',
                           attributes_dict={'size': size})

    return phantom


def run_experiment():
    phantom = get_or_create_phantom(SIZE, PHANTOM_DIR)

    rng = np.random.default_rng(SEED)
    phantom_noisy = phantom + SIGMA_NOISE_ADDED * rng.standard_normal(phantom.shape)

    denoiser = mj.QGGMRFDenoiser(phantom.shape)
    denoiser.set_params(sharpness=SHARPNESS)

    num_gpus_visible = len(jax.devices('gpu')) if jax.devices('gpu') else 0
    # shard_devices stays None on mbirjax releases before multi-GPU sharding (< v0.7.0);
    # on those releases denoising always runs on a single (the only visible) GPU.
    num_devices_used = len(denoiser.shard_devices) if denoiser.shard_devices is not None \
        else num_gpus_visible

    print('size={0}x{0}x{0}, sigma_noise={1}, num_gpus_visible={2}, num_devices_used={3}'.format(
        SIZE, SIGMA_NOISE, num_gpus_visible, num_devices_used))

    # Only the denoise() call itself is timed -- phantom load/generation happens above,
    # outside the timed region.
    t0 = time.time()
    phantom_denoised, recon_dict = denoiser.denoise(
        phantom_noisy, sigma_noise=SIGMA_NOISE, max_iterations=MAX_ITERATIONS,
        stop_threshold_change_pct=STOP_THRESHOLD_CHANGE_PCT)
    elapsed = time.time() - t0

    memory_stats = mj.get_memory_stats(print_results=False)
    gpu_stats = [entry for entry in memory_stats if entry['id'] != 'CPU']
    cpu_stats = next(entry for entry in memory_stats if entry['id'] == 'CPU')

    peak_gpu_bytes_per_device = {entry['id']: entry['peak_bytes_in_use'] for entry in gpu_stats}
    peak_gpu_bytes_total = sum(peak_gpu_bytes_per_device.values())
    peak_cpu_bytes = cpu_stats['peak_bytes_in_use']

    nrmse = np.linalg.norm(phantom_denoised - phantom) / np.linalg.norm(phantom)

    print('Elapsed denoise time = {:.3f} s'.format(elapsed))
    print('Peak GPU memory = {:.3f} GB total across {} device(s)'.format(
        peak_gpu_bytes_total / (1024 ** 3), len(gpu_stats)))
    print('Peak CPU (host) memory = {:.3f} GB'.format(peak_cpu_bytes / (1024 ** 3)))
    print('NRMSE = {:.4f}'.format(nrmse))

    return {
        'size': SIZE,
        'sigma_noise_added': SIGMA_NOISE_ADDED,
        'sigma_noise_requested': SIGMA_NOISE,
        'sharpness': SHARPNESS,
        'max_iterations': MAX_ITERATIONS,
        'num_iterations_run': recon_dict['recon_params']['num_iterations'],
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'num_gpus_visible': num_gpus_visible,
        'num_devices_used': num_devices_used,
        'elapsed_seconds': elapsed,
        'peak_gpu_gb_total': peak_gpu_bytes_total / (1024 ** 3),
        'peak_gpu_gb_per_device': json.dumps(
            {k: round(v / (1024 ** 3), 3) for k, v in peak_gpu_bytes_per_device.items()}),
        'peak_cpu_gb': peak_cpu_bytes / (1024 ** 3),
        'nrmse': nrmse,
    }


def append_result_row(row, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    file_exists = os.path.isfile(output_path)
    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        SIZE = int(sys.argv[1])
    if len(sys.argv) > 2:
        OUTPUT_CSV = sys.argv[2]
    result_row = run_experiment()
    append_result_row(result_row, OUTPUT_CSV)
    print('Appended results to {}'.format(OUTPUT_CSV))
