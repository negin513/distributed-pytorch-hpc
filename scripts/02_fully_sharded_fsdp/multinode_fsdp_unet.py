#!/usr/bin/env python3
"""
Fully Sharded Data Parallel (FSDP) training of a simple U-Net on synthetic
ERA5-like data.

Usage:
    # Single GPU (no launcher needed)
    python multinode_fsdp_unet.py

    # Single node, multiple GPUs
    torchrun --standalone --nproc_per_node=4 multinode_fsdp_unet.py

    # Multi-node with MPI (2 nodes, 4 GPUs each)
    mpiexec -n 8 --ppn 4 --cpu-bind none python multinode_fsdp_unet.py
"""
import os
import sys
import socket
import argparse
import functools
import time

# Add repo root to path so `from utils...` works
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

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
    def __init__(self, num_samples=1000, num_variables=5, num_levels=13, lat=181, lon=360):
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
# Simple U-Net
# =============================================================================

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=65, out_channels=65, base_dim=64):
        super().__init__()
        self.enc1 = self._block(in_channels, base_dim)
        self.enc2 = self._block(base_dim, base_dim * 2)
        self.enc3 = self._block(base_dim * 2, base_dim * 4)

        self.bottleneck = self._block(base_dim * 4, base_dim * 8)

        self.up3 = nn.ConvTranspose2d(base_dim * 8, base_dim * 4, 2, 2)
        self.dec3 = self._block(base_dim * 8, base_dim * 4)
        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 2, 2)
        self.dec2 = self._block(base_dim * 4, base_dim * 2)
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, 2, 2)
        self.dec1 = self._block(base_dim * 2, base_dim)

        self.out = nn.Conv2d(base_dim, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        # Interpolate upsampled tensors to match encoder skip connection sizes.
        # Required when spatial dims are odd (e.g. lat=181): MaxPool2d floors
        # 181→90→45→22, but ConvTranspose2d doubles 22→44 ≠ 45.
        up3 = torch.nn.functional.interpolate(self.up3(b), size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([up3, e3], dim=1))
        up2 = torch.nn.functional.interpolate(self.up2(d3), size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))
        up1 = torch.nn.functional.interpolate(self.up1(d2), size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))
        return self.out(d1)


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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per GPU")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of synthetic samples")
    parser.add_argument("--num_variables", type=int, default=5, help="Number of ERA5 variables")
    parser.add_argument("--num_levels", type=int, default=13, help="Number of pressure levels")
    parser.add_argument("--lat", type=int, default=181, help="Latitude grid size")
    parser.add_argument("--lon", type=int, default=360, help="Longitude grid size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    argv = parser.parse_args()

    # -----------------------------------------------------------------
    # 1. Initialize distributed (handles all launchers automatically)
    # -----------------------------------------------------------------
    WORLD_RANK, WORLD_SIZE, LOCAL_RANK = init_distributed()
    print(
        f"[{socket.gethostname()}] world_rank={WORLD_RANK} "
        f"local_rank={LOCAL_RANK} "
        f"visible_gpus={torch.cuda.device_count()}",
        flush=True,
    )
    device = torch.device(f"cuda:{LOCAL_RANK}")

    if WORLD_RANK == 0:
        print(f"World    : {WORLD_SIZE} GPUs")
        print("-" * 75)

    # -----------------------------------------------------------------
    # 2. Dataset + DistributedSampler + DataLoader
    # -----------------------------------------------------------------
    channels = argv.num_variables * argv.num_levels

    train_dataset = ERA5Dataset(
        num_samples=argv.num_samples,
        num_variables=argv.num_variables,
        num_levels=argv.num_levels,
        lat=argv.lat,
        lon=argv.lon,
    )

    # DistributedSampler partitions data across GPUs
    train_sampler = DistributedSampler(dataset=train_dataset) if WORLD_SIZE > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=argv.batch_size,
        shuffle=(train_sampler is None),   # only shuffle when NOT using DistributedSampler
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    if WORLD_RANK == 0:
        print(f"Dataset samples  : {len(train_dataset)}")
        print(f"Channels (C)     : {channels}")
        print(f"Grid (lat x lon) : {argv.lat} x {argv.lon}")
        print(f"Samples per GPU  : ~{len(train_dataset) // max(WORLD_SIZE, 1)}")
        print(f"Batches per GPU  : {len(train_loader)}")
        print(f"Effective batch  : {argv.batch_size * WORLD_SIZE}")
        print("-" * 75)

    # -----------------------------------------------------------------
    # 3. Model → GPU → FSDP
    # -----------------------------------------------------------------
    model = SimpleUNet(in_channels=channels, out_channels=channels).to(device)

    if WORLD_SIZE > 1:
        # Auto-wrap any submodule with >=100k params in its own FSDP unit so
        # parameters/grads/optim state are sharded at sub-module granularity.
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=int(1e5)
        )
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=LOCAL_RANK,
        )
    else:
        fsdp_model = model

    # -----------------------------------------------------------------
    # 4. Optimizer
    # -----------------------------------------------------------------
    optimizer = optim.AdamW(fsdp_model.parameters(), lr=argv.lr, weight_decay=argv.weight_decay)

    # -----------------------------------------------------------------
    # 5. Training loop
    # -----------------------------------------------------------------
    fsdp_model.train()
    total_start = time.perf_counter()
    total_samples_processed = 0

    for epoch in range(argv.num_epochs):
        # Ensure different shuffling each epoch
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
            output = fsdp_model(data)
            loss = latitude_weighted_mse(output, target)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_time = time.perf_counter() - batch_start

            batch_samples = data.size(0)
            epoch_samples += batch_samples
            # Global throughput: all ranks process batch_samples in parallel
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
            print("-" * 75)
            print(
                f"Epoch {epoch+1}/{argv.num_epochs} | "
                f"Avg Loss: {avg_loss:.6f} | "
                f"Time: {epoch_time:.1f}s | "
                f"Throughput: {epoch_throughput:.1f} samples/s"
            )
            print("-" * 75)

        # Checkpoint (adjust modulo as needed). FSDP requires all ranks to
        # participate in state_dict collection; FULL_STATE_DICT gathers the
        # unsharded weights onto rank 0.
        if (epoch + 1) % 1000 == 0:
            if WORLD_SIZE > 1:
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
                    state = fsdp_model.state_dict()
            else:
                state = fsdp_model.state_dict()

            if WORLD_RANK == 0:
                ckpt_path = f"checkpoint_epoch_{epoch+1}.pt"
                torch.save(state, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")
            if WORLD_SIZE > 1:
                dist.barrier()

    # -----------------------------------------------------------------
    # 6. Final throughput summary
    # -----------------------------------------------------------------
    total_time = time.perf_counter() - total_start
    if WORLD_RANK == 0:
        print("=" * 75)
        print(f"Training complete | {argv.num_epochs} epochs | {WORLD_SIZE} GPUs")
        print(f"Total time        : {total_time:.1f}s")
        print(f"Total samples     : {total_samples_processed}")
        print(f"Avg throughput    : {total_samples_processed / total_time:.1f} samples/s")
        print(f"Per-GPU throughput: {total_samples_processed / total_time / WORLD_SIZE:.1f} samples/s/GPU")
        print("=" * 75)

    # -----------------------------------------------------------------
    # 7. Cleanup
    # -----------------------------------------------------------------
    cleanup_distributed()


if __name__ == "__main__":
    main()
