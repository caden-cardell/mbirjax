import hashlib
import numpy as np
import time
import jax.numpy as jnp
import mbirjax as mj
import jax

# sinogram shape
num_views = 2000
num_det_rows = 2000
num_det_channels = 2000
sinogram_shape = (num_views, num_det_rows, num_det_channels)

# angles
start_angle = -np.pi * (1/2)
end_angle = np.pi * (1/2)
angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

# recon model
back_projection_model = mj.ParallelBeamModel(sinogram_shape, angles, use_gpu='sharding')
sinogram = jnp.ones_like(2, shape=sinogram_shape, device=back_projection_model.sinogram_device)

# Print out model parameters
back_projection_model.print_params()

# get subset pixel indices
recon_shape, _ = back_projection_model.get_params(['recon_shape', 'granularity'])
pixel_indices = mj.gen_full_indices(recon_shape)
pixel_indices = jax.device_put(pixel_indices, device=back_projection_model.worker)

############################### SHARDED ###############################
print("Starting sharded back projection")

mj.get_memory_stats()
time0 = time.time()
sharded_back_projection = back_projection_model.sparse_back_project(sinogram, pixel_indices,
                                                                    output_device=back_projection_model.main_device)
sharded_back_projection.block_until_ready()
elapsed = time.time() - time0
mj.get_memory_stats()
print('Elapsed time for back projection is {:.3f} seconds'.format(elapsed))

# view back projection
recon_rows, recon_cols, recon_slices = recon_shape
pixel_indices = jax.device_put(pixel_indices, back_projection_model.main_device)
sharded_back_projection = jax.device_put(sharded_back_projection, back_projection_model.main_device)
row_index, col_index = jnp.unravel_index(pixel_indices, recon_shape[:2])
recon = jnp.zeros(recon_shape, device=back_projection_model.main_device)
recon = recon.at[row_index, col_index].set(sharded_back_projection)
mj.slice_viewer(recon, slice_axis=2, title='Sharded Back Projection')

# save back projection
# sharded_back_projection = np.array(sharded_back_projection)
# hash_digest = hashlib.sha256(sharded_back_projection.tobytes()).hexdigest()
# file_path = f"output/sharded_back_projection_{hash_digest[:8]}.npy"
# print(f"Sharded back projection being saved to file {file_path}")
# np.save(file_path, sharded_back_projection)
