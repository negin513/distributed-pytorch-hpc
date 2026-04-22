"""
Basic Tensor Parallelism — Megatron-style Column + Row Parallel MLP

Demonstrates the core idea of tensor parallelism:
  - ColwiseParallel: split weight columns across GPUs, each GPU computes
    a slice of the output, then all-gather to reassemble.
  - RowwiseParallel: split weight rows across GPUs, each GPU computes a
    partial sum, then all-reduce to combine.

Pairing them (column -> activation -> row) lets the intermediate
activation stay distributed — only one all-reduce per MLP block.

Uses a 1D DeviceMesh (all GPUs in one TP group, no data parallelism).

Example run (single node, 4 GPUs):
    torchrun --standalone --nproc_per_node=4 02_basic_tensor_parallel.py
    mpiexec -n 4 --ppn 4 --cpu-bind none python 02_basic_tensor_parallel.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

from utils.distributed import init_distributed, cleanup_distributed
from utils.logging import get_logger, rank_log

logger = get_logger()

rank, world_size, local_rank = init_distributed()
device = torch.device("cuda", local_rank)

# ── Model ────────────────────────────────────────────────────────────
# A simple 2-layer MLP.  Hidden dim must be divisible by world_size
# so the weight can be evenly split across GPUs.
class ToyMLP(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=64, out_dim=16):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)    # Column-parallel
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(hidden_dim, out_dim)  # Row-parallel

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))

# ── Device Mesh (1D) ────────────────────────────────────────────────
# All GPUs belong to a single tensor-parallel group.
mesh = init_device_mesh("cuda", (world_size,))
rank_log(rank, logger, f"1D mesh: {world_size} GPUs in one TP group")

# ── Apply Tensor Parallelism ────────────────────────────────────────
model = ToyMLP().to(device)

model = parallelize_module(
    module=model,
    device_mesh=mesh,
    parallelize_plan={
        "in_proj":  ColwiseParallel(),   # shard weight columns
        "out_proj": RowwiseParallel(),   # shard weight rows
    },
)

rank_log(rank, logger, "Model parallelized (ColwiseParallel -> RowwiseParallel)")

# ── Training Loop ───────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
num_iters = 5

for i in range(num_iters):
    torch.manual_seed(i)                     # same input on every rank
    inp = torch.randn(8, 16, device=device)  # (batch, in_dim)
    loss = model(inp).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    rank_log(rank, logger, f"  iter {i}  loss={loss.item():.4f}")

rank_log(rank, logger, "Basic TP training complete.")

# ── Cleanup ──────────────────────────────────────────────────────────
dist.barrier()
cleanup_distributed()
