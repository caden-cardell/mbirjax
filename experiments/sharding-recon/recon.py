# -*- coding: utf-8 -*-
"""
Time only the recon() call for a cone-beam problem, using data cached on disk.

Generating the cone-beam projection data (forward-projecting a large Shepp-Logan phantom) is
itself non-trivial work, and it would otherwise be repeated -- and add noise to the timing --
on every run of a scaling / GPU-count sweep. This script caches, for a given sinogram shape,
the clean phantom + synthetic sinogram + model params as an HDF5 file the first time that
shape is asked for, and loads them from disk on every later call. So data generation / IO
happens once per shape and is excluded from the timed region: only recon_model.recon() is timed.

Following denoiser.py in this directory, a single recon() call is timed (it therefore includes
one-time JIT compilation, which is representative of the end-user experience). Multi-GPU
sharding is selected automatically from the visible devices (use_gpu='automatic'), so the
number of GPUs used is controlled entirely by CUDA_VISIBLE_DEVICES / the SLURM --gpus-per-node
request -- no code change is needed to go from 1 to 2 to 4 GPUs.

Reproducibility: recon()'s pixel partitions are drawn from numpy's global RNG, so we seed it
(SEED) before every timed call to keep the reconstruction -- and therefore NRMSE -- comparable
across GPU counts and repeats.

Usage:
    python recon.py                                  # uses the defaults below
    python recon.py <views> <rows> <channels>        # sets the sinogram shape
    python recon.py <views> <rows> <channels> <csv>  # also sets the output CSV path
"""

import csv
import datetime
import json
import os
import socket
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import h5py
import numpy as np
import jax
import mbirjax as mj
from mbirjax import generate_3d_shepp_logan_low_dynamic_range

# ignore the TkAgg warning to clean up the output
import warnings
warnings.filterwarnings("ignore", message="TkAgg not available. Falling back to Agg.")

# ##########################
# Edit these to change the run
NUM_VIEWS = 904          # sinogram axis 0 (projection angles)
NUM_DET_ROWS = 1496      # sinogram axis 1 (detector rows -> recon slices; the sharded axis)
NUM_DET_CHANNELS = 1800  # sinogram axis 2 (detector channels -> recon rows/cols)
DATA_DIR = '/scratch/gautschi/ncardel/recon_mem'  # Holds cached cone_<V>_<R>_<C>_projection_data.h5
WEIGHT_TYPE = 'transmission_root'  # Passed to mj.gen_weights; set to None for uniform weights.
STOP_THRESHOLD_CHANGE_PCT = 0.0    # 0 guarantees exactly MAX_ITERATIONS (comparable timing).
MAX_ITERATIONS = 15
SEED = 42  # Seed for recon's pixel partitions, so repeats/GPU-counts are comparable.
OUTPUT_CSV = 'logs/recon_mem.txt'  # CSV file to append this run's results to.
# ##########################


def _cache_path(num_views, num_det_rows, num_det_channels):
    return os.path.join(
        DATA_DIR, 'cone_{}_{}_{}_projection_data.h5'.format(num_views, num_det_rows, num_det_channels))


def create_recon_data(num_views, num_det_rows, num_det_channels):
    """Generate and cache the clean phantom, synthetic sinogram, and model params for this shape.

    No-op if the cache file already exists. This is the recon analog of denoiser.py's
    get_or_create_phantom -- it keeps all generation/IO out of the timed region.
    """
    h5_path = _cache_path(num_views, num_det_rows, num_det_channels)
    if os.path.isfile(h5_path):
        print('{} already exists for {}, {}, {} data'.format(
            h5_path, num_views, num_det_rows, num_det_channels))
        return
    print('creating {}, {}, {} data'.format(num_views, num_det_rows, num_det_channels))

    os.makedirs(DATA_DIR, exist_ok=True)

    start_angle = -np.pi
    end_angle = np.pi
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = np.linspace(start_angle, end_angle, num_views, endpoint=False)

    source_detector_dist = 4 * num_det_channels
    source_iso_dist = source_detector_dist
    ct_model_for_generation = mj.ConeBeamModel(sinogram_shape, angles,
                                               source_detector_dist=source_detector_dist,
                                               source_iso_dist=source_iso_dist)
    ct_model_for_generation.set_params(use_gpu='projections')

    print('Creating phantom')
    recon_shape = ct_model_for_generation.get_params('recon_shape')
    phantom = generate_3d_shepp_logan_low_dynamic_range(recon_shape)

    print('Creating sinogram')
    sinogram = ct_model_for_generation.forward_project(phantom)
    sinogram = np.asarray(sinogram)

    # Cache the arrays plus the handful of geometry params needed to rebuild the ConeBeamModel.
    # (We store geometry explicitly rather than serializing the model, so the cache is robust to
    # mbirjax model-serialization API changes.)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('phantom', data=np.asarray(phantom))
        f.create_dataset('sinogram', data=sinogram)
        f.create_dataset('angles', data=np.asarray(angles))
        f.attrs['source_detector_dist'] = source_detector_dist
        f.attrs['source_iso_dist'] = source_iso_dist


def _sweep_axis(num_views, num_det_rows, num_det_channels, base=512):
    """Label which single axis is being swept above `base` (for grouping in the report)."""
    changed = [name for name, val in
               (('views', num_views), ('rows', num_det_rows), ('channels', num_det_channels))
               if val != base]
    if not changed:
        return 'baseline'
    if len(changed) == 1:
        return changed[0]
    return 'mixed'


def _mbirjax_version():
    try:
        from importlib.metadata import version
        return version('mbirjax')
    except Exception:
        return getattr(mj, '__version__', 'unknown')


def _visible_gpus():
    """List of visible GPU devices, or [] on a CPU-only backend (jax.devices('gpu') raises there)."""
    try:
        return list(jax.devices('gpu'))
    except RuntimeError:
        return []


def _gpu_name():
    gpus = _visible_gpus()
    return gpus[0].device_kind if gpus else 'cpu'


def run_experiment(num_views, num_det_rows, num_det_channels):
    h5_path = _cache_path(num_views, num_det_rows, num_det_channels)
    with h5py.File(h5_path, 'r') as f:
        sinogram = f['sinogram'][:]
        phantom = f['phantom'][:]
        angles = f['angles'][:]
        source_detector_dist = float(f.attrs['source_detector_dist'])
        source_iso_dist = float(f.attrs['source_iso_dist'])

    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    recon_model = mj.ConeBeamModel(sinogram_shape, angles,
                                   source_detector_dist=source_detector_dist,
                                   source_iso_dist=source_iso_dist)
    recon_model.set_params(use_gpu='automatic')

    num_gpus_visible = len(_visible_gpus())
    num_devices_used = len(recon_model.shard_devices) if recon_model.shard_devices is not None \
        else num_gpus_visible

    print('shape=({0},{1},{2}), num_gpus_visible={3}, num_devices_used={4}'.format(
        num_views, num_det_rows, num_det_channels, num_gpus_visible, num_devices_used))

    if WEIGHT_TYPE is None:
        weights = None
    else:
        weights = mj.gen_weights(sinogram / sinogram.max(), weight_type=WEIGHT_TYPE)

    print('\nGPU STARTING MEMORY STATS:')
    mj.get_memory_stats()

    # Only the recon() call itself is timed -- data load / weight gen happen above, outside the
    # timed region. Seed numpy so the (stochastic) pixel partitions are reproducible.
    np.random.seed(SEED)
    t0 = time.time()
    recon, recon_dict = recon_model.recon(
        sinogram, weights=weights, max_iterations=MAX_ITERATIONS,
        stop_threshold_change_pct=STOP_THRESHOLD_CHANGE_PCT)
    # recon() returns a materialized numpy array by default (output_sharded=False), which is
    # already host-synchronized; block only if it handed back a still-async jax array.
    if hasattr(recon, 'block_until_ready'):
        recon.block_until_ready()
    elapsed = time.time() - t0

    memory_stats = mj.get_memory_stats(print_results=False)
    gpu_stats = [entry for entry in memory_stats if entry['id'] != 'CPU']
    cpu_stats = next(entry for entry in memory_stats if entry['id'] == 'CPU')

    peak_gpu_bytes_per_device = {entry['id']: entry['peak_bytes_in_use'] for entry in gpu_stats}
    peak_gpu_bytes_total = sum(peak_gpu_bytes_per_device.values())
    peak_cpu_bytes = cpu_stats['peak_bytes_in_use']

    recon = np.asarray(recon)
    nrmse = float(np.linalg.norm(recon - phantom) / np.linalg.norm(phantom))
    num_iterations_run = recon_dict['recon_params']['num_iterations']

    print('\nELAPSED TIME: {:.3f} seconds'.format(elapsed))
    print('Peak GPU memory = {:.3f} GB total across {} device(s)'.format(
        peak_gpu_bytes_total / (1024 ** 3), len(gpu_stats)))
    print('Peak CPU (host) memory = {:.3f} GB'.format(peak_cpu_bytes / (1024 ** 3)))
    print('NRMSE = {:.4f}'.format(nrmse))

    return {
        'num_views': num_views,
        'num_det_rows': num_det_rows,
        'num_det_channels': num_det_channels,
        'sweep_axis': _sweep_axis(num_views, num_det_rows, num_det_channels),
        'num_recon_voxels': int(np.prod(recon.shape)),
        'num_sino_elements': int(num_views * num_det_rows * num_det_channels),
        'max_iterations': MAX_ITERATIONS,
        'stop_threshold_change_pct': STOP_THRESHOLD_CHANGE_PCT,
        'num_iterations_run': num_iterations_run,
        'seed': SEED,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'num_gpus_visible': num_gpus_visible,
        'num_devices_used': num_devices_used,
        'elapsed_seconds': elapsed,
        'sec_per_iteration': elapsed / num_iterations_run if num_iterations_run else float('nan'),
        'peak_gpu_gb_total': peak_gpu_bytes_total / (1024 ** 3),
        'peak_gpu_gb_per_device': json.dumps(
            {k: round(v / (1024 ** 3), 3) for k, v in peak_gpu_bytes_per_device.items()}),
        'peak_cpu_gb': peak_cpu_bytes / (1024 ** 3),
        'nrmse': nrmse,
        'mbirjax_version': _mbirjax_version(),
        'git_commit': os.environ.get('GIT_COMMIT', ''),
        'hostname': socket.gethostname(),
        'gpu_name': _gpu_name(),
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
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
    if len(sys.argv) > 3:
        NUM_VIEWS = int(sys.argv[1])
        NUM_DET_ROWS = int(sys.argv[2])
        NUM_DET_CHANNELS = int(sys.argv[3])
    if len(sys.argv) > 4:
        OUTPUT_CSV = sys.argv[4]

    create_recon_data(NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS)
    result_row = run_experiment(NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS)
    append_result_row(result_row, OUTPUT_CSV)
    print('Appended results to {}'.format(OUTPUT_CSV))
