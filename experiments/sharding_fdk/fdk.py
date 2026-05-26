import sys
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import time
import csv
import h5py
import mbirjax as mj
import jax
import numpy as np
from mbirjax import generate_3d_shepp_logan_low_dynamic_range

# ignore the TkAgg warning to clean up the output
import warnings
warnings.filterwarnings(
    "ignore",
    message="TkAgg not available. Falling back to Agg."
)

def create_fdk_data(num_views, num_det_rows, num_det_channels):

    output_directory = f"/scratch/gautschi/ncardel/recon_mem"
    h5_path = f"{output_directory}/cone_{num_views}_{num_det_rows}_{num_det_channels}_projection_data.h5"
    if os.path.isfile(h5_path):
        print(f"{h5_path} already exists for {num_views}, {num_det_rows}, {num_det_channels} data")
        return
    print(f"creating {num_views}, {num_det_rows}, {num_det_channels} data")


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

    # Generate phantom
    print('Creating phantom')
    recon_shape = ct_model_for_generation.get_params('recon_shape')
    device = ct_model_for_generation.main_device
    phantom = generate_3d_shepp_logan_low_dynamic_range(recon_shape, device=device)

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = ct_model_for_generation.forward_project(phantom)
    sinogram = np.asarray(sinogram)

    # save the phantom, sinogram, and params
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("phantom", data=phantom)
        f.create_dataset("sinogram", data=sinogram)
        f.attrs["params"] = ct_model_for_generation.to_file(None)


def fdk(num_views, num_det_rows, num_det_channels, output_filepath='output.csv'):

    output_directory = f"/scratch/gautschi/ncardel/recon_mem"
    h5_path = f"{output_directory}/cone_{num_views}_{num_det_rows}_{num_det_channels}_projection_data.h5"
    with h5py.File(h5_path, "r") as f:
        sinogram = f["sinogram"][:]
        params = f.attrs["params"]

    filter_model = mj.ConeBeamModel.from_file(params)

    print("\nTEST PARAMS:")
    transfer_pixel_batch_size = filter_model.transfer_pixel_batch_size
    print("Transfer pixel batch size:", transfer_pixel_batch_size)
    try:
        print("Device set:", filter_model.sinogram_device.device_set)
    except:
        pass

    print("\nGPU STARTING MEMORY STATS:")
    mj.get_memory_stats()

    print("\nSTARTING FDK:")
    filter_model.set_params(use_gpu="automatic")
    time0 = time.time()
    sinogram = jax.device_put(sinogram, device=filter_model.sinogram_device)
    filtered_sinogram = filter_model.fdk_filter(sinogram)
    filtered_sinogram.block_until_ready()

    elapsed = time.time() - time0

    print('\nELAPSED TIME: {:.3f} seconds'.format(elapsed))

    print("\nGPU FINAL MEMORY STATS:")
    mem_stats = mj.get_memory_stats()

    num_gpus = 4
    gpu_col_names = [f'gpu{i}_peak_bytes' for i in range(num_gpus)]

    # if the output file doesn't exist then create it
    print("output_filepath:", output_filepath)
    os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)
    if not os.path.exists(output_filepath):
        with open(output_filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['num_views', 'num_det_rows', 'num_det_channels', 'elapsed_seconds', 'transfer_pixel_batch_size'] + gpu_col_names)

    # append this test data to the output file
    row = [num_views, num_det_rows, num_det_channels, round(elapsed, 3), filter_model.transfer_pixel_batch_size] + [mem_stats[i]['peak_bytes_in_use'] for i in range(min(num_gpus, len(mem_stats)))]
    with open(output_filepath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

if __name__ == "__main__":

    try:
        num_views = int(sys.argv[1])
        num_det_rows = int(sys.argv[2])
        num_det_channels = int(sys.argv[3])
        output_filepath = sys.argv[4]
    except:
        num_views = 2048
        num_det_rows = 2048
        num_det_channels = 2048
        output_filepath = "logs/recon_mem.txt"

    create_fdk_data(num_views, num_det_rows, num_det_channels)
    fdk(num_views, num_det_rows, num_det_channels, output_filepath=output_filepath)