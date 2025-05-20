import jax
import numpy as np
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P

arr = np.arange(32).reshape(4, 8)
print(arr)

# create a device mesh
devices = np.array(jax.devices('gpu')).reshape(2, 4)
mesh = Mesh(devices, ('x', 'y'))

# create a sharding
sharding = NamedSharding(mesh, P('x', 'y'))
arr_sharded = jax.device_put(arr, sharding)

# create a partially replicated sharding
replicating = NamedSharding(mesh, P(None, 'y'))
arr_replicated = jax.device_put(arr, replicating)

# visualize the sharded array
jax.debug.visualize_array_sharding(arr_sharded)
print(arr_sharded.addressable_shards)

# visualize the partially replicated array
jax.debug.visualize_array_sharding(arr_replicated)
print(arr_replicated.addressable_shards)