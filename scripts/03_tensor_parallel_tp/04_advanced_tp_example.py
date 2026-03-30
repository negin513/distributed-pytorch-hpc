"""
Tensor Parallel (TP) demo on a toy model using PyTorch native DTensor APIs.
This runs in a Megatron-LM–style setup (column/row parallel MLP)
with forward, backward, and optimization steps.

Example run (Derecho): 2 nodes x 4 gpus/node
    mpiexec -n 8 --ppn 4 --cpu-bind none python tensor_parallel_example.py
or 1 node x 4 gpus/node
    mpiexec -n 4 --ppn 4 --cpu-bind none python tensor_parallel_example.py
    torchrun --nproc_per_node=4 tensor_parallel_example.py
"""

import os
import sys

# Add repo root to path so `from utils...` works
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

from utils.distributed import init_distributed, cleanup_distributed
from utils.logging import get_logger, rank_log, verify_min_gpu_count

logger = get_logger()

if not verify_min_gpu_count(4):
    print("This example requires at least 4 GPUs per node.")
    sys.exit(1)

# Initialize distributed (supports torchrun, mpiexec, Cray MPICH)
rank, world_size, local_rank = init_distributed()
device = torch.device("cuda", local_rank)

print(f"[Rank {rank}] using cuda:{local_rank} (world_size={world_size})")

# ---------------------------------------------------------------------
# Define toy MLP model
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(64, 64)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))

# ---------------------------------------------------------------------
mesh_1d=True
if mesh_1d:
    # Create device mesh
    assert world_size % 4 == 0, f"Need even world_size, got {world_size}"

    # Mesh shape product must equal world_size
    # 1-D Mesh shape
    mesh_shape = (world_size,)
    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=mesh_shape)
    rank_log(rank, logger, f"Device mesh created: {device_mesh}")
    dist.barrier()
else:
    # ---------------------------------------------------------------------
    # Create 2-D device mesh: Tensor Parallel (TP) × Data Parallel (DP)
    tp = 4                                   # number of GPUs per tensor-parallel group
    assert world_size % tp == 0, f"World size {world_size} not divisible by TP={tp}"
    dp = world_size // tp                    # number of data-parallel groups
    mesh_shape = (tp, dp)
    print (mesh_shape)
    device_type="cuda"
    device_mesh = init_device_mesh(device_type, (dp, tp), mesh_dim_names=("dp", "tp"))

    rank_log(rank, logger, f"2-D Device mesh created with shape {device_mesh.mesh.shape}")
    dist.barrier()
# ---------------------------------------------------------------------
# Build model and apply tensor parallelism
tp_model = ToyModel().to(device)

tp_model = parallelize_module(
    module=tp_model,
    device_mesh=device_mesh,
    parallelize_plan={
        "in_proj": ColwiseParallel(),
        "out_proj": RowwiseParallel(),
    },
)

# optimizer
optimizer = torch.optim.AdamW(tp_model.parameters(), lr=0.25, foreach=True)

# ---------------------------------------------------------------------
# Run training loop
num_iters = 5

for i in range(num_iters):
    torch.manual_seed(i)
    inp = torch.randn(64, 64, device=device)
    out = tp_model(inp)
    out.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    rank_log(rank, logger, f"Iteration {i} complete")

rank_log(rank, logger, "Tensor Parallel training finished!")

# ---------------------------------------------------------------------
# Clean teardown
dist.barrier()
cleanup_distributed()

