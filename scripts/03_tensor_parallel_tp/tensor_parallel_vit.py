#!/usr/bin/env python3
"""
Tensor-Parallel Vision Transformer (ViT) on Synthetic ERA5-like Data

Demonstrates how tensor parallelism shards Transformer layers across GPUs,
using the same Megatron-LM style parallelism that powers production LLM
training (GPT, LLaMA, etc.).

What gets tensor-parallelized (sharded across GPUs within a node):
  - Attention Q, K, V projections → ColwiseParallel (each GPU gets a
    subset of attention heads)
  - Attention output projection   → RowwiseParallel (all-reduce combines)
  - MLP first linear (fc1)        → ColwiseParallel (each GPU gets a
    slice of hidden dim)
  - MLP second linear (fc2)       → RowwiseParallel (all-reduce combines)

Uses a 2D DeviceMesh: TP within each node (fast NVLink), DP across nodes.

Example run:
    # Single node, 4 GPUs (tp=4, dp=1)
    torchrun --standalone --nproc_per_node=4 tensor_parallel_vit.py
    mpiexec -n 4 --ppn 4 --cpu-bind none python tensor_parallel_vit.py

    # Multi-node, 2 nodes x 4 GPUs (tp=4, dp=2)
    mpiexec -n 8 --ppn 4 --cpu-bind none python tensor_parallel_vit.py
"""
import os
import sys
import socket
import argparse
import time

# Add repo root to path so `from utils...` works
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

from utils.distributed import init_distributed, cleanup_distributed


# =============================================================================
# ERA5-like Dataset
# =============================================================================

class ERA5Dataset(Dataset):
    """
    Simulates ERA5-like data for weather prediction.

    Each sample is a pair (input, target) of shape (channels, lat, lon)
    where channels = num_variables * num_levels.
    """
    def __init__(self, num_samples=500, num_variables=5, num_levels=13, lat=64, lon=128):
        self.num_samples = num_samples
        self.channels = num_variables * num_levels
        self.lat = lat
        self.lon = lon

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(self.channels, self.lat, self.lon)
        y = torch.randn(self.channels, self.lat, self.lon)
        return x, y


# =============================================================================
# Vision Transformer (ViT) for Weather Prediction
# =============================================================================

class PatchEmbed(nn.Module):
    """Convert (B, C, H, W) image into (B, num_patches, embed_dim) tokens."""
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """
    Multi-head self-attention with separate Q, K, V projections.

    Separate projections (instead of a combined QKV) give clean TP sharding:
    ColwiseParallel on q/k/v gives each GPU a disjoint subset of attention
    heads; RowwiseParallel on out_proj all-reduces the partial results.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)   # → ColwiseParallel
        self.k_proj = nn.Linear(dim, dim)   # → ColwiseParallel
        self.v_proj = nn.Linear(dim, dim)   # → ColwiseParallel
        self.out_proj = nn.Linear(dim, dim) # → RowwiseParallel

    def forward(self, x):
        B, N, _ = x.shape
        # After TP sharding, each GPU holds num_heads/tp_size heads.
        # Using -1 lets both TP and non-TP paths reshape correctly.
        q = self.q_proj(x).reshape(B, N, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, -1, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(x)


class MLP(nn.Module):
    """Feed-forward: fc1 (ColwiseParallel) → GELU → fc2 (RowwiseParallel)."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleViT(nn.Module):
    """
    Vision Transformer for weather prediction (regression).

    Patch-embeds the input, processes with Transformer blocks, then
    projects back to the original spatial grid.
    """
    def __init__(self, in_channels, out_channels, patch_size,
                 lat, lon, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.h_patches = lat // patch_size
        self.w_patches = lon // patch_size
        num_patches = self.h_patches * self.w_patches

        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, embed_dim) * 0.02
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads,
                             mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Project back to pixel space: embed_dim → C * patch_h * patch_w
        self.head = nn.Linear(embed_dim,
                              out_channels * patch_size * patch_size)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) + self.pos_embed   # (B, num_patches, D)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x)   # (B, num_patches, C*P*P)

        # Reshape patches back to spatial grid
        x = x.reshape(B, self.h_patches, self.w_patches,
                       self.out_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(
            B, self.out_channels,
            self.h_patches * self.patch_size,
            self.w_patches * self.patch_size,
        )
        return x


# =============================================================================
# Latitude-Weighted MSE Loss
# =============================================================================

def latitude_weighted_mse(pred, target):
    """
    MSE weighted by cosine of latitude so that grid cells near the equator
    (which represent more area) contribute more to the loss.
    """
    lat = torch.linspace(90, -90, pred.shape[-2], device=pred.device)
    weights = torch.cos(torch.deg2rad(lat)).view(1, 1, -1, 1)
    weights = weights / weights.mean()
    return (weights * (pred - target) ** 2).mean()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tensor-Parallel ViT on ERA5-like data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size per GPU")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of synthetic samples")
    parser.add_argument("--num_variables", type=int, default=5,
                        help="Number of ERA5 variables")
    parser.add_argument("--num_levels", type=int, default=13,
                        help="Number of pressure levels")
    parser.add_argument("--lat", type=int, default=64,
                        help="Latitude grid size (must be divisible by patch_size)")
    parser.add_argument("--lon", type=int, default=128,
                        help="Longitude grid size (must be divisible by patch_size)")
    parser.add_argument("--patch_size", type=int, default=8,
                        help="Patch size for ViT tokenization")
    parser.add_argument("--embed_dim", type=int, default=256,
                        help="Embedding dimension (must be divisible by tp_size)")
    parser.add_argument("--depth", type=int, default=6,
                        help="Number of Transformer blocks")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Attention heads (must be divisible by tp_size)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    argv = parser.parse_args()

    # -----------------------------------------------------------------
    # 1. Initialize distributed (handles all launchers automatically)
    # -----------------------------------------------------------------
    WORLD_RANK, WORLD_SIZE, LOCAL_RANK = init_distributed()
    device = torch.device(f"cuda:{LOCAL_RANK}")

    print(
        f"[{socket.gethostname()}] world_rank={WORLD_RANK} "
        f"local_rank={LOCAL_RANK} "
        f"visible_gpus={torch.cuda.device_count()}",
        flush=True,
    )

    # -----------------------------------------------------------------
    # 2. Device Mesh (2D: dp x tp)
    # -----------------------------------------------------------------
    tp_size = min(4, WORLD_SIZE)
    dp_size = WORLD_SIZE // tp_size

    device_mesh = init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )
    tp_mesh = device_mesh["tp"]

    if WORLD_RANK == 0:
        print(f"{'=' * 75}")
        print(f"Tensor-Parallel ViT Training on ERA5-like Data")
        print(f"{'=' * 75}")
        print(f"  World size     : {WORLD_SIZE} GPUs")
        print(f"  2D Mesh        : dp={dp_size} x tp={tp_size}")
        print(f"  Mesh tensor    :\n{device_mesh.mesh}")
        print(f"{'-' * 75}")

    # -----------------------------------------------------------------
    # 3. Dataset + DistributedSampler + DataLoader
    # -----------------------------------------------------------------
    channels = argv.num_variables * argv.num_levels

    train_dataset = ERA5Dataset(
        num_samples=argv.num_samples,
        num_variables=argv.num_variables,
        num_levels=argv.num_levels,
        lat=argv.lat,
        lon=argv.lon,
    )

    train_sampler = DistributedSampler(
        dataset=train_dataset
    ) if WORLD_SIZE > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=argv.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    if WORLD_RANK == 0:
        num_patches = (argv.lat // argv.patch_size) * \
                      (argv.lon // argv.patch_size)
        print(f"  Dataset samples: {len(train_dataset)}")
        print(f"  Channels (C)   : {channels}")
        print(f"  Grid (lat x lon): {argv.lat} x {argv.lon}")
        print(f"  Patch size     : {argv.patch_size} -> {num_patches} patches")
        print(f"  Embed dim      : {argv.embed_dim}")
        print(f"  Depth          : {argv.depth} transformer blocks")
        print(f"  Heads          : {argv.num_heads}")
        print(f"  Samples per GPU: ~{len(train_dataset) // max(WORLD_SIZE, 1)}")
        print(f"  Batches per GPU: {len(train_loader)}")
        print(f"  Effective batch: {argv.batch_size * WORLD_SIZE}")
        print(f"{'-' * 75}")

    # -----------------------------------------------------------------
    # 4. Model -> GPU -> Tensor Parallel
    # -----------------------------------------------------------------
    model = SimpleViT(
        in_channels=channels,
        out_channels=channels,
        patch_size=argv.patch_size,
        lat=argv.lat,
        lon=argv.lon,
        embed_dim=argv.embed_dim,
        depth=argv.depth,
        num_heads=argv.num_heads,
    ).to(device)

    # Build the TP plan: shard attention and MLP linears in every block.
    #
    #   ColwiseParallel splits weight columns -> each GPU computes a slice
    #   RowwiseParallel splits weight rows    -> all-reduce combines slices
    #
    #   Attention: Q,K,V (Colwise) -> local head computation -> O (Rowwise)
    #   MLP:       fc1   (Colwise) -> GELU                  -> fc2 (Rowwise)
    tp_plan = {}
    for i in range(argv.depth):
        tp_plan[f"blocks.{i}.attn.q_proj"] = ColwiseParallel()
        tp_plan[f"blocks.{i}.attn.k_proj"] = ColwiseParallel()
        tp_plan[f"blocks.{i}.attn.v_proj"] = ColwiseParallel()
        tp_plan[f"blocks.{i}.attn.out_proj"] = RowwiseParallel()
        tp_plan[f"blocks.{i}.mlp.fc1"] = ColwiseParallel()
        tp_plan[f"blocks.{i}.mlp.fc2"] = RowwiseParallel()

    model = parallelize_module(model, tp_mesh, tp_plan)

    if WORLD_RANK == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Model params (per GPU, after TP sharding): {param_count:,}")
        print(f"  TP layers      : {len(tp_plan)} linears parallelized")
        print(f"{'-' * 75}")

    # -----------------------------------------------------------------
    # 5. Optimizer
    # -----------------------------------------------------------------
    # foreach=False required: AdamW's _foreach ops can't mix regular
    # tensors (patch_embed, pos_embed, norms) with DTensors (TP-sharded
    # linears) in a single vectorised update.
    optimizer = optim.AdamW(
        model.parameters(), lr=argv.lr, weight_decay=argv.weight_decay,
        foreach=False,
    )

    # -----------------------------------------------------------------
    # 6. Training loop
    # -----------------------------------------------------------------
    model.train()
    total_start = time.perf_counter()
    total_samples_processed = 0

    for epoch in range(argv.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_start = time.perf_counter()
        epoch_samples = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            batch_start = time.perf_counter()
            optimizer.zero_grad()
            output = model(data)
            loss = latitude_weighted_mse(output, target)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            batch_time = time.perf_counter() - batch_start

            batch_samples = data.size(0)
            epoch_samples += batch_samples
            global_throughput = (batch_samples * WORLD_SIZE) / batch_time
            epoch_loss += loss.item()

            if batch_idx % 10 == 0 and WORLD_RANK == 0:
                print(
                    f"[Rank {WORLD_RANK}] Epoch {epoch+1}, "
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.6f}, "
                    f"Throughput: {global_throughput:.1f} samples/s"
                )

        epoch_time = time.perf_counter() - epoch_start
        total_samples_processed += epoch_samples * WORLD_SIZE
        epoch_throughput = (epoch_samples * WORLD_SIZE) / epoch_time
        avg_loss = epoch_loss / len(train_loader)

        if WORLD_RANK == 0:
            print(f"{'-' * 75}")
            print(
                f"Epoch {epoch+1}/{argv.num_epochs} | "
                f"Avg Loss: {avg_loss:.6f} | "
                f"Time: {epoch_time:.1f}s | "
                f"Throughput: {epoch_throughput:.1f} samples/s"
            )
            print(f"{'-' * 75}")

    # -----------------------------------------------------------------
    # 7. Final throughput summary
    # -----------------------------------------------------------------
    total_time = time.perf_counter() - total_start
    if WORLD_RANK == 0:
        print(f"{'=' * 75}")
        print(f"Training complete | {argv.num_epochs} epochs | "
              f"{WORLD_SIZE} GPUs (tp={tp_size}, dp={dp_size})")
        print(f"Total time        : {total_time:.1f}s")
        print(f"Total samples     : {total_samples_processed}")
        print(f"Avg throughput    : "
              f"{total_samples_processed / total_time:.1f} samples/s")
        print(f"Per-GPU throughput: "
              f"{total_samples_processed / total_time / WORLD_SIZE:.1f} "
              f"samples/s/GPU")
        print(f"{'=' * 75}")

    # -----------------------------------------------------------------
    # 8. Cleanup
    # -----------------------------------------------------------------
    cleanup_distributed()


if __name__ == "__main__":
    main()
