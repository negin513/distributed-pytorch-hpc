"""
Tensor Parallel (TP) demo on a toy model using PyTorch native DTensor APIs.
This runs in a Megatron-LM–style setup (column/row parallel MLP)
with forward, backward, and optimization steps.

Example run (Derecho): 2 nodes x 4 gpus/node
    mpiexec -n 8 --ppn 4 --cpu-bind none python tensor_parallel_example.py
or 1node x 4 gpus/node
    torchrun --nproc_per_node=4 tensor_parallel_example.py
"""

import os
import sys
import socket
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from mpi4py import MPI
from torch.distributed._tensor.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

# ---------------------------------------------------------------------
# Logging setup
logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO
)
logger = logging.getLogger(__name__)

def rank_log(rank, msg):
    if rank == 0:
        logger.info(msg)

# ---------------------------------------------------------------------
# Basic GPU validation
def verify_min_gpu_count(min_gpus: int = 4) -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= min_gpus

if not verify_min_gpu_count(4):
    print("This example requires at least 4 GPUs per node.")
    sys.exit(1)

# ---------------------------------------------------------------------
# Initialize MPI + torch.distributed
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    shmem_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    local_rank = shmem_comm.Get_rank()
    local_size = shmem_comm.Get_size()
except:
    import sys
    sys.exit("Can't find the evironment variables for local rank")


# Share master address and port across all ranks
os.environ["MASTER_ADDR"] = comm.bcast(socket.gethostbyname(socket.gethostname()), root=0)
os.environ["MASTER_PORT"] = "12355"

dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# Assign one GPU per local rank
torch.cuda.set_device(local_rank % torch.cuda.device_count())
device = torch.device("cuda", local_rank % torch.cuda.device_count())

print(f"[Rank {rank}] using cuda:{local_rank} (local_size={local_size}, world_size={world_size})")

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
    rank_log(rank, f"Device mesh created: {device_mesh}")
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

    rank_log(rank, f"2-D Device mesh created with shape {device_mesh.mesh.shape}")
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
    rank_log(rank, f"Iteration {i} complete")

rank_log(rank, "Tensor Parallel training finished!")

# ---------------------------------------------------------------------
# Clean teardown
dist.barrier()
dist.destroy_process_group()

