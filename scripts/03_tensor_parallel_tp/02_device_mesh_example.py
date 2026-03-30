import os
import sys

# Add repo root to path so `from utils...` works
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from utils.distributed import init_distributed, cleanup_distributed

rank, world_size, local_rank = init_distributed()

# ------- FIX: choose a mesh whose product == world_size -------
# Example 1D mesh:
# mesh_shape = (world_size,)
# mesh_dim_names = ("dp",)

# Example 2D mesh (e.g., tp x dp):
tp = 2
assert world_size % tp == 0, "world_size must be divisible by tp"
dp = world_size // tp
mesh_shape = (tp, dp)
mesh_dim_names = ("tp", "dp")

device_mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=mesh_shape,
    mesh_dim_names=mesh_dim_names,
)

# … your DTensor / parallel code …

# Clean teardown to avoid the warning about ProcessGroupNCCL not destroyed
dist.barrier()
cleanup_distributed()

