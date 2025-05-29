import os
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import time
from functools import partial
from pprint import pprint as pp

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.98'

def set_sinogram_parameters():
    # Specify sinogram info
    num_views = 2000
    num_det_rows = 500
    num_det_channels = 500
    return num_views, num_det_rows, num_det_channels

def main():
    """
    This is a script to develop, debug, and tune the parallel beam projector
    """

    output_device = jax.devices('cpu')[0]
    try:
        # Get available gpu devices and create a mesh
        devices = np.array(jax.devices('gpu')).reshape((-1, 1))
        mesh = Mesh(devices, ('views', 'rows'))
        sharded_worker = NamedSharding(mesh, P('views'))
        replicated_worker = NamedSharding(mesh, P())
    except RuntimeError:
        # this is a GPU test so raise an error if anything fails with the GPU
        raise RuntimeError("GPU failed")

    num_views, num_det_rows, num_det_channels = set_sinogram_parameters()

    # force single batch
    max_views_per_batch = 200
    max_pixels_per_batch = int(num_det_channels * num_det_channels // 128)
    num_pixels_to_exclude = 0

    sinogram = []
    for i in range(num_views // max_views_per_batch):
        cur_view_batch = jnp.full([max_views_per_batch, num_det_rows, num_det_channels], i, device=sharded_worker)
        sinogram.append(cur_view_batch)

    sinogram = jnp.concatenate(sinogram)
    return sinogram

if __name__ == "__main__":
    main()
