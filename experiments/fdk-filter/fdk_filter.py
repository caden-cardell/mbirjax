import sys, os, time
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax.numpy as jnp
import mbirjax as mj

def fdk(sinogram_shape):

    num_view, _, num_channels = sinogram_shape

    # params
    start_angle = -jnp.pi / 2
    end_angle = jnp.pi / 2

    angles = jnp.linspace(start_angle, end_angle, num_view, endpoint=False)
    source_detector_dist = 4 * num_channels
    source_iso_dist = source_detector_dist

    # create model
    fdk_model = mj.ConeBeamModel(sinogram_shape, angles,
                                 source_detector_dist=source_detector_dist,
                                 source_iso_dist=source_iso_dist)

    # create sinogram
    print("\nsinogram shape:", sinogram_shape)
    sinogram = jnp.full(sinogram_shape, 2)

    # perform filtering
    time0 = time.time()
    fdk_model.fdk_filter(sinogram)
    elapsed_time = time.time() - time0
    print(f"elapsed_time: {elapsed_time:.4f} seconds")

    # get memory usage stats for filtering
    memory_stats_per_processor = mj.get_memory_stats(print_results=False)
    gb = 1024 ** 3
    peak_gpu_memory_usage_bytes = next(s['peak_bytes_in_use'] for s in memory_stats_per_processor if s['id'] == 'GPU 0')
    print(f"peak_memory_usage: {peak_gpu_memory_usage_bytes / gb:.4f}  GiB")

    return elapsed_time, peak_gpu_memory_usage_bytes


if __name__ == "__main__":

    try:
        num_views = int(sys.argv[1])
        num_det_rows = int(sys.argv[2])
        num_det_channels = int(sys.argv[3])

        fdk((num_views, num_det_rows, num_det_channels))

    except:

        for size in range(256, 65_536, 256):

            try:
                fdk((size, 1024, 1024))
            except:
                print(f"Failed as size: {size}")
                exit(0)