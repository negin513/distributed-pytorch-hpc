"""
DeviceMesh Basics — Organizing GPUs for Parallel Strategies

DeviceMesh is PyTorch's abstraction for arranging GPUs into logical
topologies.  Every parallelism strategy in this directory builds on it,
so understanding meshes first makes everything else easier.

Example run (single node, 4 GPUs):
    torchrun --standalone --nproc_per_node=4 01_device_mesh_basics.py
    mpiexec -n 4 --ppn 4 --cpu-bind none python 01_device_mesh_basics.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from utils.distributed import init_distributed, cleanup_distributed

rank, world_size, local_rank = init_distributed()
device = torch.device("cuda", local_rank)

# ── 1D Mesh ──────────────────────────────────────────────────────────
# All GPUs in a single flat group — used for plain TP or plain DP.
mesh_1d = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))

if rank == 0:
    print("=" * 60)
    print("1D Mesh (all GPUs in one group)")
    print(f"  Shape : {mesh_1d.mesh.shape}")
    print(f"  Tensor: {mesh_1d.mesh}")
    print()

dist.barrier()

# ── 2D Mesh ──────────────────────────────────────────────────────────
# Two dimensions: TP (tensor parallel) inside each node, DP (data
# parallel) across nodes.  The LAST dimension is the fastest-varying
# one, so GPUs in the same TP group sit on the same node.
#
# Example with 8 GPUs (2 nodes x 4 GPUs):
#   mesh shape (dp=2, tp=4) produces:
#       TP group 0: [GPU 0, 1, 2, 3]   (node 0)
#       TP group 1: [GPU 4, 5, 6, 7]   (node 1)
#       DP group 0: [GPU 0, 4]
#       DP group 1: [GPU 1, 5]  ...etc

tp_size = min(4, world_size)
dp_size = world_size // tp_size

if world_size >= 4 and world_size % tp_size == 0:
    mesh_2d = init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )

    if rank == 0:
        print("2D Mesh (dp x tp)")
        print(f"  Shape : {mesh_2d.mesh.shape}")
        print(f"  Tensor:\n{mesh_2d.mesh}")
        print()

    # Slice out sub-meshes — this is how you pass the right group to
    # parallelize_module() or FSDP.
    tp_mesh = mesh_2d["tp"]   # 1D mesh for the TP dimension
    dp_mesh = mesh_2d["dp"]   # 1D mesh for the DP dimension

    if rank == 0:
        print(f"  TP sub-mesh (rank 0's group): {tp_mesh.mesh}")
        print(f"  DP sub-mesh (rank 0's group): {dp_mesh.mesh}")
        print()
else:
    if rank == 0:
        print(f"Skipping 2D mesh (need world_size >= 4, got {world_size})")

dist.barrier()

# ── Quick communication test ─────────────────────────────────────────
tensor = torch.tensor([rank], dtype=torch.float32, device=device)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
expected = sum(range(world_size))
if rank == 0:
    print(f"all_reduce sanity check: result={tensor.item():.0f}, expected={expected}")
    print("=" * 60)

dist.barrier()
cleanup_distributed()
