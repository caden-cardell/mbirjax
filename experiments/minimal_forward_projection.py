import os
import numpy as np
import jax.numpy as jnp
import jax
import time
from functools import partial
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import json
import psutil

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.98'

"""
Forward projection is creating a sinogram, in this case from a phantom.

we want to "scan" and create a sinogram of the shape defined in `set_sinogram_parameters` views*rows*channels using a 
phantom that is channels*channels*rows

Our phantom (recon) is channels*channels*rows*mem_per_entry gb
Our sinogram is views*rows*channels*mem_per_entry gb

Its a straight shot from one to the other
"""

def append_to_json_list_file(path: str, data: dict):
    """
    Reads the existing JSON list (or starts one), appends data, and writes it back.
    """
    try:
        with open(path, 'r+') as f:
            records = json.load(f)
            if not isinstance(records, list):
                raise ValueError("JSON root is not a list")
            records.append(data)
            f.seek(0)
            json.dump(records, f, indent=2)
            f.truncate()
    except (FileNotFoundError, json.JSONDecodeError):
        # file missing or empty/invalid: start fresh
        with open(path, 'w') as f:
            json.dump([data], f, indent=2)

def set_sinogram_parameters():
    # Specify sinogram info
    num_views = 200
    num_det_rows = 150
    num_det_channels = 400
    return num_views, num_det_rows, num_det_channels

def set_batch_parameters():
    max_views_per_batch = 200
    max_pixels_per_batch = 800
    num_pixels_to_exclude = 100
    return max_views_per_batch, max_pixels_per_batch, num_pixels_to_exclude

def sparse_forward_project(voxel_values, indices, sinogram_shape, recon_shape, angles, output_device, sharded_worker, replicated_worker):
    """
    Batch the views (angles) and voxels/indices, send batches to the GPU to project, and collect the results.
    """
    max_views_per_batch, max_pixels_per_batch, num_pixels_to_exclude = set_batch_parameters()

    indices = indices[:len(indices)-num_pixels_to_exclude]
    angles = jax.device_put(angles, device=sharded_worker)

    # TODO: print debug
    jax.debug.visualize_array_sharding(angles)

    # Batch the views and pixels
    num_views = len(angles)
    view_batch_indices = jnp.arange(num_views, step=max_views_per_batch)
    view_batch_indices = jnp.concatenate([view_batch_indices, num_views * jnp.ones(1, dtype=int)])

    num_pixels = len(indices)
    pixel_batch_indices = jnp.arange(num_pixels, step=max_pixels_per_batch)
    pixel_batch_indices = jnp.concatenate([pixel_batch_indices, num_pixels * jnp.ones(1, dtype=int)])

    # Create the output sinogram
    sinogram = []

    # Loop over the view batches
    for j, view_index_start in enumerate(view_batch_indices[:-1]):
        # Send a batch of views to worker
        view_index_end = view_batch_indices[j+1]
        cur_view_batch = jnp.zeros([view_index_end-view_index_start, sinogram_shape[1], sinogram_shape[2]],
                                   device=sharded_worker)
        cur_view_params_batch = angles[view_index_start:view_index_end]
        if j == 0:
            get_memory_stats()
        print('Starting view block {} of {}.'.format(j+1, view_batch_indices.shape[0]-1))

        # TODO: look at set devices and batch sizes for setting the parameters

        # In sparse forward project in the tomography_model.py file on line
        # first hope that sharding with sinogram_views with minimal changes (line 404)

        # Loop over pixel batches
        for k, pixel_index_start in enumerate(pixel_batch_indices[:-1]):
            # Send a batch of pixels to worker
            pixel_index_end = pixel_batch_indices[k+1]
            cur_voxel_batch = jax.device_put(voxel_values[pixel_index_start:pixel_index_end], replicated_worker)
            cur_index_batch = jax.device_put(indices[pixel_index_start:pixel_index_end], replicated_worker)

            if len(cur_index_batch) < max_pixels_per_batch:
                cur_voxel_batch = jnp.concatenate([cur_voxel_batch, jnp.zeros([max_pixels_per_batch-len(cur_index_batch), sinogram_shape[1],], device=replicated_worker)])
                cur_index_batch = jnp.concatenate([cur_index_batch, jnp.zeros([max_pixels_per_batch-len(cur_index_batch)], device=replicated_worker)])

            def forward_project_pixel_batch_local(view, angle):
                # Add the forward projection to the given existing view
                return forward_project_pixel_batch_to_one_view(cur_voxel_batch, cur_index_batch, angle, view,
                                                               sinogram_shape, recon_shape)

            view_map = jax.vmap(forward_project_pixel_batch_local)
            # print(jax.make_jaxpr(view_map)(cur_view_batch, cur_view_params_batch))
            # input('Enter to continue')
            # a = jax.jit(view_map).lower(cur_view_batch, cur_view_params_batch).compiler_ir('hlo')
            # with open("outfile.dot", "w") as f:
            #     f.write(a.as_hlo_dot_graph())
            # dot outfile.dot  -Tpng > outfile.png
            # or
            # dot -Tps outfile.dot -o outfile.ps
            # ps2pdf outfile.ps
            # print(jax.jit(view_map).lower(cur_view_batch, cur_view_params_batch).compile().as_text())
            cur_view_batch = view_map(cur_view_batch, cur_view_params_batch)

        sinogram.append(jax.device_put(cur_view_batch, output_device))
    sinogram = jnp.concatenate(sinogram)
    return sinogram


@partial(jax.jit, static_argnames=['sinogram_shape', 'recon_shape'], donate_argnames='sinogram_view')
def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, angle, sinogram_view,
                                            sinogram_shape, recon_shape):
    """
    Apply a parallel beam transformation to a set of voxel cylinders. These cylinders are assumed to have
    slices aligned with detector rows, so that a parallel beam maps a cylinder slice to a detector row.
    This function returns the resulting sinogram view.

    """
    # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
    # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
    num_views, num_det_rows, num_det_channels = sinogram_shape
    psf_radius = 1
    delta_voxel = 1

    # Get the data needed for horizontal projection
    n_p, n_p_center, W_p_c, cos_alpha_p_xy = compute_proj_data(pixel_indices, angle, sinogram_shape, recon_shape)
    L_max = jnp.minimum(1, W_p_c)

    # Do the projection
    for n_offset in jnp.arange(start=-psf_radius, stop=psf_radius+1):
        n = n_p_center + n_offset
        abs_delta_p_c_n = jnp.abs(n_p - n)
        L_p_c_n = jnp.clip((W_p_c + 1) / 2 - abs_delta_p_c_n, 0, L_max)
        A_chan_n = delta_voxel * L_p_c_n / cos_alpha_p_xy
        A_chan_n *= (n >= 0) * (n < num_det_channels)
        sinogram_view = sinogram_view.at[:, n].add(A_chan_n.reshape((1, -1)) * voxel_values.T)

    return sinogram_view

def compute_proj_data(pixel_indices, angle, sinogram_shape, recon_shape):
    """
    Compute the quantities n_p, n_p_center, W_p_c, cos_alpha_p_xy needed for vertical projection.
    """

    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)

    delta_voxel = 1.0
    dvc = delta_voxel
    dvs = delta_voxel
    dvc *= cosine
    dvs *= sine

    num_views, num_det_rows, num_det_channels = sinogram_shape

    # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
    row_index, col_index = jnp.unravel_index(pixel_indices, recon_shape[:2])

    y_tilde = dvs * (row_index - (recon_shape[0] - 1) / 2.0)
    x_tilde = dvc * (col_index - (recon_shape[1] - 1) / 2.0)

    x_p = x_tilde - y_tilde

    det_center_channel = (num_det_channels - 1) / 2.0  # num_of_cols

    # Calculate indices on the detector grid
    n_p = x_p + det_center_channel
    n_p_center = jnp.round(n_p).astype(int)

    # Compute cos alpha for row and columns
    cos_alpha_p_xy = jnp.maximum(jnp.abs(cosine), jnp.abs(sine))

    # Compute projected voxel width along columns and rows (in fraction of detector size)
    W_p_c = cos_alpha_p_xy

    proj_data = (n_p, n_p_center, W_p_c, cos_alpha_p_xy)

    return proj_data


def get_memory_stats(print_results=True, file=None):
    # Get all GPU devices
    gpus = [device for device in jax.devices() if 'cpu' not in device.device_kind.lower()]

    # Collect memory info for gpus
    mem_stats = []
    for gpu in gpus:
        # Memory info returns total_memory and available_memory in bytes
        gpu_stats = gpu.memory_stats()
        memory_stats = dict()
        memory_stats['id'] = 'GPU ' + str(gpu.id)
        memory_stats['bytes_in_use'] = gpu_stats['bytes_in_use']
        memory_stats['peak_bytes_in_use'] = gpu_stats['peak_bytes_in_use']
        memory_stats['bytes_limit'] = gpu_stats['bytes_limit']
        mem_stats.append(gpu_stats)

        print(memory_stats['id'], file=file)
        for tag in ['bytes_in_use', 'peak_bytes_in_use', 'bytes_limit']:
            cur_value = memory_stats[tag] / (1024 ** 3)
            extra_space = ' ' * (21 - len(tag) - len(str(int(cur_value))))
            print(f'  {tag}:{extra_space}{cur_value:.3f}GB', file=file)

    return mem_stats


def set_devices_and_batch_sizes():

    # Get the cpu and any gpus
    # If no gpu, then use the cpu and return
    cpus = jax.devices('cpu')
    gb = 1024 ** 3
    use_gpu = 'automatic'
    try:
        gpus = jax.devices('gpu')
        gpu_memory_stats = gpus[0].memory_stats()
        gpu_memory = float(gpu_memory_stats['bytes_limit']) - float(gpu_memory_stats['bytes_in_use'])
        gpu_memory /= gb
    except RuntimeError:
        if use_gpu not in ['automatic', 'none']:
            raise RuntimeError("'use_gpu' is set to {} but no gpu is available.  Reset to 'automatic' or 'none'.".format(use_gpu))
        gpus = []
        gpu_memory = 0

    # Estimate the memory available and required for this problem
    pid = os.getpid()
    current_process = psutil.Process(pid)
    memory_info = current_process.memory_full_info()
    memory_stats = dict()
    memory_stats['id'] = 'CPU'
    memory_stats['peak_bytes_in_use'] = memory_info.rss
    memory_stats['bytes_in_use'] = memory_info.uss  # This is the 'Unique Set Size' used by the process
    # Get the virtual memory statistics
    mem = psutil.virtual_memory()
    memory_stats['bytes_limit'] = mem.available
    cpu_memory_stats = memory_stats
    cpu_memory = float(cpu_memory_stats['bytes_limit']) - float(cpu_memory_stats['bytes_in_use'])
    cpu_memory /= gb

    num_views, num_det_rows, num_det_channels = set_sinogram_parameters()
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    recon_shape = (num_det_channels, num_det_channels, num_det_rows)

    zero = jnp.zeros(1)
    bits_per_byte = 8
    mem_per_entry = float(str(zero.dtype)[5:]) / bits_per_byte / gb  # Parse floatXX to get the number of bits
    memory_per_sinogram = mem_per_entry * np.prod(sinogram_shape)
    memory_per_recon = mem_per_entry * np.prod(recon_shape)

    total_memory_required = memory_per_sinogram + memory_per_recon
    subset_update_memory_required = memory_per_sinogram + memory_per_recon

    # Set the default batch sizes, then adjust as needed and update memory requirements
    views_per_batch = 256 * len(jax.devices('gpu'))
    pixels_per_batch = 2048
    num_slices = max(sinogram_shape[1], recon_shape[2])
    projection_memory_per_view = pixels_per_batch * num_slices * mem_per_entry

    subset_memory_excess = gpu_memory - subset_update_memory_required
    subset_views_per_batch = subset_memory_excess // projection_memory_per_view
    subset_views_per_batch = int(np.clip(subset_views_per_batch, 2, views_per_batch))

    total_memory_excess = cpu_memory - total_memory_required
    total_views_per_batch = total_memory_excess // projection_memory_per_view
    total_views_per_batch = int(np.clip(total_views_per_batch, 2, views_per_batch))

    subset_update_memory_required += subset_views_per_batch * projection_memory_per_view
    total_memory_required += total_views_per_batch * projection_memory_per_view



def main():
    """
    This is a script to develop, debug, and tune the parallel beam projector
    """

    # Specify sinogram info
    num_views, num_det_rows, num_det_channels = set_sinogram_parameters()
    start_angle = 0
    end_angle = jnp.pi
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    output_device = jax.devices('cpu')[0]
    try:
        # Get available gpu devices and create a mesh
        devices = np.array(jax.devices('gpu'))
        mesh = Mesh(devices, 'x')
        sharded_worker = NamedSharding(mesh, P('x'))
        replicated_worker = NamedSharding(mesh, P())

        use_gpu = True
    except RuntimeError:
        raise RuntimeError("GPU failed")


    # Generate phantom - all zero except a small cube
    recon_shape = (num_det_channels, num_det_channels, num_det_rows)
    num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
    with jax.default_device(output_device):
        phantom = jnp.zeros(recon_shape)  #mbirjax.gen_cube_phantom(recon_shape)
        i, j, k = recon_shape[0]//3, recon_shape[1]//2, recon_shape[2]//2
        phantom = phantom.at[i:i+5, j:j+5, k:k+5].set(1.0)

        # Generate indices of pixels and sinogram data
        # Determine the 2D indices within the RoR
        max_index_val = num_recon_rows * num_recon_cols
        indices = np.arange(max_index_val, dtype=np.int32)
        indices = jnp.array(indices)
        voxel_values = phantom.reshape((-1,) + recon_shape[2:])[indices]

    print('Starting forward projection')
    voxel_values, indices = jax.device_put([voxel_values, indices], output_device)
    t0 = time.time()
    sinogram = sparse_forward_project(voxel_values, indices, sinogram_shape, recon_shape, angles,
                                      output_device=output_device,
                                      sharded_worker=sharded_worker,
                                      replicated_worker=replicated_worker)
    print('Elapsed time:', time.time() - t0)

    # Determine resulting number of views, slices, and channels and image size
    print('Sinogram shape: {}'.format(sinogram.shape))

    mem_stats = None
    if use_gpu:
        print('Memory stats after forward projection')
        mem_stats = get_memory_stats(print_results=True)

    num_views, num_det_rows, num_det_channels = set_sinogram_parameters()
    max_views_per_batch, max_pixels_per_batch, num_pixels_to_exclude = set_batch_parameters()

    record = {
        "params": {
            'num_views': num_views,
            'num_det_rows': num_det_rows,
            'num_det_channels': num_det_channels,
            'max_views_per_batch': max_views_per_batch,
            'max_pixels_per_batch': max_pixels_per_batch,
            'num_pixels_to_exclude': num_pixels_to_exclude
        },
        "time_sec": time.time() - t0,
        "mem_stats": mem_stats,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    append_to_json_list_file("metrics.json", record)

    # import mbirjax
    # mbirjax.slice_viewer(sinogram, slice_axis=0)


if __name__ == "__main__":
    main()
    # set_devices_and_batch_sizes()