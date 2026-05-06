#!/usr/bin/env python3
"""
04_domain_parallel_with_fsdp.py - Domain parallelism with manual halo exchange + FSDP

Implements domain parallelism from scratch using PyTorch distributed primitives.
No PhysicsNeMo required - just PyTorch + NCCL.

This demonstrates the SAME concept as PhysicsNeMo's ShardTensor, but built
explicitly so you can see exactly what happens:
    1. Each GPU holds a spatial slice of the input (height dimension)
    2. Before each convolution, neighboring GPUs exchange "halo" rows
    3. Each GPU pads its local slice, runs the conv, then trims the padding
    4. FSDP handles weight sharding across the data-parallel dimension

Run with mpiexec:
    mpiexec -n 4 --ppn 4 --cpu-bind none python 04_domain_parallel_with_fsdp.py
    mpiexec -n 4 --ppn 4 --cpu-bind none python 04_domain_parallel_with_fsdp.py --domain-size 2 --fsdp-size 2

Run with torchrun:
    torchrun --standalone --nproc_per_node=4 04_domain_parallel_with_fsdp.py
    torchrun --standalone --nproc_per_node=4 04_domain_parallel_with_fsdp.py --domain-size 2 --fsdp-size 2

The 2D mesh (for domain=2, fsdp=2):
              domain_dim ->
    fsdp_dim  [GPU0, GPU1]     <- GPU0 & GPU1 each hold half the image height
       |      [GPU2, GPU3]     <- GPU2 & GPU3 each hold half the image height
       v
    GPU0 & GPU2 share weights via FSDP (data-parallel replicas)
"""

import os
import sys
import time
import argparse

# Add repo root to path so `from utils...` works
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from utils.distributed import init_distributed, cleanup_distributed


# ============================================================
# Halo Exchange: the core of domain decomposition
# ============================================================
class HaloExchange(torch.autograd.Function):
    """
    Exchange halo (ghost) rows between neighboring GPUs.

    For a 3x3 convolution with padding=1, each GPU needs 1 extra row
    from each neighbor. This function:
      - Sends the top row to the neighbor above
      - Sends the bottom row to the neighbor below
      - Receives halos and pads the local tensor

    The backward pass reverses the communication: gradients from halo
    regions are sent back to the GPU that owns those rows.

    Args:
        x: local tensor [B, C, H_local, W]
        halo_size: number of rows to exchange (typically kernel_size // 2)
        domain_group: process group for domain-parallel GPUs
        prev_global_rank: global rank of the neighbor above (or -1 if none)
        next_global_rank: global rank of the neighbor below (or -1 if none)
    """

    @staticmethod
    def forward(ctx, x, halo_size, domain_group, prev_global_rank, next_global_rank):
        ctx.halo_size = halo_size
        ctx.domain_group = domain_group
        ctx.prev_global_rank = prev_global_rank
        ctx.next_global_rank = next_global_rank

        has_prev = prev_global_rank >= 0
        has_next = next_global_rank >= 0

        B, C, H, W = x.shape
        device = x.device

        # Buffers for received halos
        recv_top = torch.zeros(B, C, halo_size, W, device=device) if has_prev else None
        recv_bot = torch.zeros(B, C, halo_size, W, device=device) if has_next else None

        ops = []

        if has_prev:
            # Send my top rows to neighbor above, receive their bottom rows
            ops.append(dist.P2POp(dist.isend, x[:, :, :halo_size, :].contiguous(), prev_global_rank, group=domain_group))
            ops.append(dist.P2POp(dist.irecv, recv_top, prev_global_rank, group=domain_group))

        if has_next:
            # Send my bottom rows to neighbor below, receive their top rows
            ops.append(dist.P2POp(dist.isend, x[:, :, -halo_size:, :].contiguous(), next_global_rank, group=domain_group))
            ops.append(dist.P2POp(dist.irecv, recv_bot, next_global_rank, group=domain_group))

        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # Pad the local tensor with received halos or zeros at boundaries.
        # Boundary GPUs (first/last in domain group) get zero-padding on the
        # side where they have no neighbor, matching standard Conv2d behavior.
        # This ensures ALL GPUs produce [B, C, H_local + 2*halo, W].
        chunks = []
        if has_prev:
            chunks.append(recv_top)
        else:
            chunks.append(torch.zeros(B, C, halo_size, W, device=device))
        chunks.append(x)
        if has_next:
            chunks.append(recv_bot)
        else:
            chunks.append(torch.zeros(B, C, halo_size, W, device=device))

        return torch.cat(chunks, dim=2)

    @staticmethod
    def backward(ctx, grad_output):
        halo_size = ctx.halo_size
        domain_group = ctx.domain_group
        prev_global_rank = ctx.prev_global_rank
        next_global_rank = ctx.next_global_rank

        has_prev = prev_global_rank >= 0
        has_next = next_global_rank >= 0

        # Strip halo gradients and send them back to their owners.
        # The padded output always has: [top_pad | local | bot_pad]
        # where each pad is halo_size (from neighbor or zero-padding).
        top_offset = halo_size
        H_padded = grad_output.shape[2]
        bot_offset = H_padded - halo_size

        # Local gradient (without halo regions)
        grad_local = grad_output[:, :, top_offset:bot_offset, :].contiguous()

        B, C, _, W = grad_local.shape
        device = grad_local.device

        recv_from_above = torch.zeros(B, C, halo_size, W, device=device) if has_prev else None
        recv_from_below = torch.zeros(B, C, halo_size, W, device=device) if has_next else None

        ops = []

        if has_prev:
            grad_top_halo = grad_output[:, :, :halo_size, :].contiguous()
            ops.append(dist.P2POp(dist.isend, grad_top_halo, prev_global_rank, group=domain_group))
            ops.append(dist.P2POp(dist.irecv, recv_from_above, prev_global_rank, group=domain_group))

        if has_next:
            grad_bot_halo = grad_output[:, :, bot_offset:, :].contiguous()
            ops.append(dist.P2POp(dist.isend, grad_bot_halo, next_global_rank, group=domain_group))
            ops.append(dist.P2POp(dist.irecv, recv_from_below, next_global_rank, group=domain_group))

        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # Add halo gradient contributions to the local gradient
        if has_prev:
            grad_local[:, :, :halo_size, :] += recv_from_above
        if has_next:
            grad_local[:, :, -halo_size:, :] += recv_from_below

        return grad_local, None, None, None, None


def halo_exchange(x, halo_size, domain_group, prev_global_rank, next_global_rank):
    """Functional wrapper for the halo exchange autograd function."""
    return HaloExchange.apply(x, halo_size, domain_group, prev_global_rank, next_global_rank)


# ============================================================
# Domain-parallel Conv2d: conv with halo exchange
# ============================================================
class DomainParallelConv2d(nn.Module):
    """
    A Conv2d that works on spatially-sharded data.

    Before applying the convolution, it performs halo exchange with
    neighbors to get the boundary rows needed for correct results.
    After the convolution, the halo padding is stripped so the output
    has the same local spatial size as the input.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 domain_group=None, prev_global_rank=-1, next_global_rank=-1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=1, padding=0)  # no padding - we do it via halo
        self.halo_size = kernel_size // 2
        self.w_padding = padding  # padding for width (non-sharded) dimension
        self.domain_group = domain_group
        self.prev_global_rank = prev_global_rank
        self.next_global_rank = next_global_rank

    def forward(self, x):
        # Step 1: Halo exchange along height (sharded dimension)
        # This adds halo_size rows on top and bottom from neighbors
        x = halo_exchange(x, self.halo_size, self.domain_group,
                          self.prev_global_rank, self.next_global_rank)

        # Step 2: Pad width dimension normally (not sharded)
        # x is now [B, C, H_local + 2*halo, W]
        # We need padding on width: [left, right, top, bottom]
        x = F.pad(x, [self.w_padding, self.w_padding, 0, 0])

        # Step 3: Apply convolution (no built-in padding)
        x = self.conv(x)
        return x


# ============================================================
# Domain-parallel model
# ============================================================
class DomainParallelCNN(nn.Module):
    """
    A CNN that uses domain-parallel convolutions with explicit halo exchange.

    GroupNorm is used instead of BatchNorm because:
    - BatchNorm computes mean/var over (B, H, W) - requires allreduce across
      domain GPUs for each layer (expensive)
    - GroupNorm computes stats over (C_group, H, W) per-sample - works locally
      on each GPU's spatial slice without communication
    """
    def __init__(self, in_channels=3, num_classes=10,
                 domain_group=None, prev_global_rank=-1, next_global_rank=-1):
        super().__init__()

        def make_conv(in_c, out_c):
            return DomainParallelConv2d(
                in_c, out_c, kernel_size=3, padding=1,
                domain_group=domain_group,
                prev_global_rank=prev_global_rank,
                next_global_rank=next_global_rank,
            )

        self.block1 = nn.Sequential(
            make_conv(in_channels, 64),
            nn.GroupNorm(8, 64),    # 8 groups over 64 channels
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            make_conv(64, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            make_conv(128, 256),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            make_conv(256, 256),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)           # [B, 256, 1, 1] - local reduction
        x = x.view(x.size(0), -1)  # [B, 256]
        x = self.classifier(x)     # [B, num_classes]
        return x


def main():
    parser = argparse.ArgumentParser(description="Domain Parallel + FSDP Training")
    parser.add_argument("--domain-size", type=int, default=None,
                        help="Number of GPUs for domain parallelism (default: all GPUs)")
    parser.add_argument("--fsdp-size", type=int, default=1,
                        help="Number of GPUs for FSDP/data parallelism (default: 1)")
    parser.add_argument("--image-size", type=int, default=1024,
                        help="Image height and width")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Global batch size (split across FSDP ranks)")
    parser.add_argument("--num-steps", type=int, default=15,
                        help="Training steps")
    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed(verbose=False)
    device = torch.device(f"cuda:{local_rank}")

    # Default: all GPUs for domain parallelism
    if args.domain_size is None:
        args.domain_size = world_size

    # Validate config
    if args.domain_size * args.fsdp_size != world_size:
        if rank == 0:
            print(f"ERROR: domain_size ({args.domain_size}) * fsdp_size ({args.fsdp_size}) "
                  f"must equal world_size ({world_size})")
        cleanup_distributed()
        sys.exit(1)

    if args.batch_size % args.fsdp_size != 0:
        if rank == 0:
            print(f"ERROR: batch_size ({args.batch_size}) must be divisible by "
                  f"fsdp_size ({args.fsdp_size})")
        cleanup_distributed()
        sys.exit(1)

    if args.image_size % args.domain_size != 0:
        if rank == 0:
            print(f"ERROR: image_size ({args.image_size}) must be divisible by "
                  f"domain_size ({args.domain_size})")
        cleanup_distributed()
        sys.exit(1)

    # ===========================================================
    # STEP 1: Create process groups for the 2D mesh
    # ===========================================================
    # Organize GPUs into a 2D grid: [fsdp_size x domain_size]
    #
    # For 4 GPUs, fsdp=2, domain=2:
    #   Global ranks laid out as:
    #     [0, 1]    <- domain group 0 (FSDP replica 0)
    #     [2, 3]    <- domain group 1 (FSDP replica 1)
    #
    #   FSDP groups (columns): {0,2}, {1,3}
    #   Domain groups (rows):  {0,1}, {2,3}

    # Which row/column am I in?
    fsdp_rank = rank // args.domain_size    # my row index
    domain_rank = rank % args.domain_size   # my column index

    # Create domain groups (one per FSDP replica)
    domain_group = None
    for f in range(args.fsdp_size):
        ranks = list(range(f * args.domain_size, (f + 1) * args.domain_size))
        group = dist.new_group(ranks)
        if fsdp_rank == f:
            domain_group = group

    # Create FSDP groups (one per domain shard)
    fsdp_group = None
    for d in range(args.domain_size):
        ranks = [f * args.domain_size + d for f in range(args.fsdp_size)]
        group = dist.new_group(ranks)
        if domain_rank == d:
            fsdp_group = group

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Domain Parallel + FSDP Training")
        print(f"{'='*60}")
        print(f"  Total GPUs:      {world_size}")
        print(f"  Domain size:     {args.domain_size} (spatial sharding)")
        print(f"  FSDP size:       {args.fsdp_size} (weight sharding)")
        print(f"  Mesh layout:     {args.fsdp_size} x {args.domain_size}")
        print(f"  Image size:      {args.image_size}x{args.image_size}")
        print(f"  Global batch:    {args.batch_size}")
        print(f"  Per-FSDP batch:  {args.batch_size // args.fsdp_size}")
        print(f"  Local H per GPU: {args.image_size // args.domain_size}")
        print(f"{'='*60}\n")

    # ===========================================================
    # STEP 2: Create local data slices
    # ===========================================================
    local_batch = args.batch_size // args.fsdp_size
    local_h = args.image_size // args.domain_size

    # Each GPU generates only its spatial slice - no full image ever created!
    # In real code, you'd load/slice your data here.
    local_images = torch.randn(
        local_batch, 3, local_h, args.image_size, device=device
    )
    local_labels = torch.randint(0, 10, (local_batch,), device=device)

    print(f"  Rank {rank} (domain_rank={domain_rank}, fsdp_rank={fsdp_rank}): "
          f"local data = {list(local_images.shape)}")

    # ===========================================================
    # STEP 3: Create model with domain-parallel convolutions
    # ===========================================================
    # Compute global ranks of neighbors in the domain group
    # Domain group for fsdp_rank f contains global ranks:
    #   [f * domain_size, f * domain_size + 1, ..., f * domain_size + domain_size - 1]
    prev_global = fsdp_rank * args.domain_size + (domain_rank - 1) if domain_rank > 0 else -1
    next_global = fsdp_rank * args.domain_size + (domain_rank + 1) if domain_rank < args.domain_size - 1 else -1

    model = DomainParallelCNN(
        in_channels=3,
        num_classes=10,
        domain_group=domain_group,
        prev_global_rank=prev_global,
        next_global_rank=next_global,
    ).to(device)

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {param_count:,}")

    # Wrap with FSDP if using data parallelism
    if args.fsdp_size > 1:
        model = FSDP(model, process_group=fsdp_group)
        if rank == 0:
            print(f"  Model wrapped with FSDP (process_group size={args.fsdp_size})\n")
    else:
        if rank == 0:
            print(f"  Pure domain parallelism (no FSDP)\n")

    # ===========================================================
    # STEP 4: Training loop
    # ===========================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    times = []

    for step in range(args.num_steps):
        start = time.time()

        optimizer.zero_grad()
        output = model(local_images)

        # AdaptiveAvgPool2d reduces each GPU's local spatial slice to 1x1,
        # but the mean is only over the LOCAL slice, not the global image.
        # For correct global mean, we'd need an allreduce across domain GPUs.
        # For training (where we just need a consistent loss), this works.
        loss = criterion(output, local_labels)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.time() - start

        if step >= 3:
            times.append(elapsed)

        if rank == 0:
            print(f"  Step {step:3d}  loss={loss.item():.4f}  time={elapsed:.3f}s")

    # ===========================================================
    # Results
    # ===========================================================
    if rank == 0 and times:
        avg_time = sum(times) / len(times)
        full_mem = args.batch_size * 3 * args.image_size * args.image_size * 4 / 1e6
        local_mem = local_batch * 3 * local_h * args.image_size * 4 / 1e6
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"  Avg step time (post-warmup): {avg_time:.3f}s")
        print(f"  Parallelism: {args.domain_size}-way domain x {args.fsdp_size}-way FSDP")
        print(f"  Input per GPU: [{local_batch}, 3, {local_h}, {args.image_size}] ({local_mem:.1f} MB)")
        print(f"  vs full image: [{args.batch_size}, 3, {args.image_size}, {args.image_size}] ({full_mem:.1f} MB)")
        print(f"  Memory reduction: {full_mem/local_mem:.1f}x per GPU")
        print(f"{'='*60}\n")

    cleanup_distributed()


if __name__ == "__main__":
    main()
