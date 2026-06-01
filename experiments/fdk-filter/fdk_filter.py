import sys, os, time
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import mbirjax as mj

output_filename = "logs/fdk_filter_results.csv"

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
    print("sinogram shape:", sinogram_shape)
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
    except IndexError:
        num_views = 256
        num_det_rows = 256
        num_det_channels = 256
        print(f"WARNING! Missing params defaulting to sinogram shape ({num_views}, {num_det_rows}, {num_det_channels})")

    # if output file doesn't exist create it and add header line
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    if not os.path.exists(output_filename):
        with open(output_filename, "a") as f:
            f.write("views,rows,channels,time,bytes\n")

    # run filter and record an OOM error if it occurs
    try:
        elapsed_time, peak_bytes = fdk((num_views, num_det_rows, num_det_channels))
    except jax.errors.JaxRuntimeError as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            print(f"Out of memory for shape ({num_views}, {num_det_rows}, {num_det_channels})")
            with open(output_filename, "a") as f:
                f.write(f"{num_views},{num_det_rows},{num_det_channels},0,0\n")
            raise e
        raise

    # save successful filter to file
    with open(output_filename, "a") as f:
        f.write(f"{num_views},{num_det_rows},{num_det_channels},{elapsed_time},{peak_bytes}\n")

