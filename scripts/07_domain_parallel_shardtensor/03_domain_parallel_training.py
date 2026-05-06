#!/usr/bin/env python3
"""
03_domain_parallel_training.py - Full training loop with domain parallelism

Trains a multi-layer CNN on synthetic high-resolution data using PyTorch's
DTensor for domain parallelism. No PhysicsNeMo required.

The input images are spatially sharded across GPUs so each GPU only holds
a slice of the height dimension.

Run with:
    mpiexec -n 4 --ppn 4 --cpu-bind none python 03_domain_parallel_training.py
    torchrun --standalone --nproc_per_node=4 03_domain_parallel_training.py

Architecture:
    A simple 3-layer CNN with group norm, ReLU, and adaptive pooling.
    The spatial layers (conv, groupnorm) operate on sharded data.
    The final pooling + linear layer operates after reducing spatial dims.

What is domain parallelism good for?
    - When batch_size=1 ALREADY exceeds GPU memory (common in weather/climate,
      medical imaging, computational physics with high-res grids)
    - When your model has many spatial layers (convolutions, norms, pooling)
    - When you have fast GPU-GPU interconnect (NVLink, InfiniBand)

What is it NOT good for?
    - Small inputs that fit on one GPU -> use DDP instead (more efficient)
    - Models dominated by tiny kernels -> overhead of halo exchange hurts
"""

import os
import sys
import time

# Add repo root to path so `from utils...` works
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module

from utils.distributed import init_distributed, cleanup_distributed


# ============================================================
# Model: A simple CNN that works with domain parallelism
# ============================================================
class DomainParallelCNN(nn.Module):
    """
    A 3-layer CNN designed for high-resolution spatial inputs.

    All spatial operations (Conv2d, GroupNorm, ReLU) are compatible
    with DTensor domain parallelism. The AdaptiveAvgPool2d at the end
    reduces the spatial dimensions to 1x1, and the final Linear layer
    operates on the channel dimension only.

    GroupNorm is used instead of BatchNorm because BatchNorm computes
    mean/var over (B, H, W) — when H is sharded across GPUs, BatchNorm
    produces incorrect statistics. GroupNorm computes stats over
    (C_group, H, W) per-sample, so it works correctly on each GPU's
    local spatial slice without cross-GPU communication.

    NOTE: Operations like x.view() or x.reshape() that flatten spatial
    dims are NOT compatible with DTensor sharding. Use AdaptiveAvgPool2d
    to reduce spatial dims first, then flatten the channel dim.
    """
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)          # [B, 128, H_local, W] - sharded spatially
        x = self.pool(x)              # [B, 128, 1, 1]   - reduces sharded dim
        x = x.view(x.size(0), -1)    # [B, 128]          - safe after pool
        x = self.classifier(x)        # [B, num_classes]
        return x


def main():
    rank, world_size, local_rank = init_distributed(verbose=False)
    device = torch.device(f"cuda:{local_rank}")

    # ----------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------
    NUM_CLASSES = 10
    IMAGE_SIZE = 1024       # 1024x1024 - large enough to benefit from sharding
    IN_CHANNELS = 3
    BATCH_SIZE = 2          # Per-GPU batch size (small because images are huge)
    NUM_STEPS = 20
    LR = 0.001

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Domain Parallel Training with DTensor")
        print(f"{'='*60}")
        print(f"  GPUs:        {world_size}")
        print(f"  Image size:  {IMAGE_SIZE}x{IMAGE_SIZE}")
        print(f"  Channels:    {IN_CHANNELS}")
        print(f"  Batch size:  {BATCH_SIZE} (per GPU)")
        print(f"  Steps:       {NUM_STEPS}")
        print(f"{'='*60}\n")

    # ===========================================================
    # STEP 1: Create the DeviceMesh
    # ===========================================================
    # All GPUs form a single domain-parallel group.
    # With 4 GPUs, the image height (1024) is split 4 ways -> 256 rows each.
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("domain",))

    # ===========================================================
    # STEP 2: Create synthetic data and distribute across GPUs
    # ===========================================================
    full_images = torch.randn(
        BATCH_SIZE, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device,
    )
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=device)

    # Shard images along height (dim=2)
    # After this, each GPU holds [BATCH_SIZE, 3, IMAGE_SIZE/world_size, IMAGE_SIZE]
    sharded_images = distribute_tensor(full_images, mesh, placements=[Shard(2)])

    # Labels are replicated (every GPU needs the full label vector for loss)
    replicated_labels = distribute_tensor(labels, mesh, placements=[Replicate()])

    local_h = sharded_images.to_local().shape[2]
    print(f"  Rank {rank}: local image shape = "
          f"[{BATCH_SIZE}, {IN_CHANNELS}, {local_h}, {IMAGE_SIZE}]")

    # ===========================================================
    # STEP 3: Create model and distribute it
    # ===========================================================
    model = DomainParallelCNN(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
    ).to(device)

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\n  Model parameters: {param_count:,}")

    # distribute_module wraps the model so it can process DTensors.
    # Weights stay replicated on all GPUs; only the activations are sharded.
    model = distribute_module(model, device_mesh=mesh)

    # ===========================================================
    # STEP 4: Training loop
    # ===========================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    if rank == 0:
        print(f"\n  Starting training...\n")

    model.train()
    times = []

    for step in range(NUM_STEPS):
        start = time.time()

        optimizer.zero_grad()

        # Forward pass - DTensor handles all halo exchange internally
        output = model(sharded_images)

        # Loss computation
        loss = criterion(output, replicated_labels)

        # Backward pass - gradients flow correctly through sharded ops
        loss.backward()

        # Optimizer step - updates replicated weights (same on all GPUs)
        optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.time() - start

        if step >= 3:  # skip warmup steps
            times.append(elapsed)

        if rank == 0:
            print(f"  Step {step:3d}  loss={loss.item():.4f}  time={elapsed:.3f}s")

    # ===========================================================
    # STEP 5: Report results
    # ===========================================================
    if rank == 0 and times:
        avg_time = sum(times) / len(times)
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"  Average step time (excluding warmup): {avg_time:.3f}s")
        print(f"  GPUs used: {world_size}")
        print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
        print(f"  Memory per GPU: ~{local_h * IMAGE_SIZE * IN_CHANNELS * 4 / 1e6:.1f} MB input "
              f"(vs {IMAGE_SIZE * IMAGE_SIZE * IN_CHANNELS * 4 / 1e6:.1f} MB full)")
        print(f"{'='*60}\n")

    cleanup_distributed()


if __name__ == "__main__":
    main()
