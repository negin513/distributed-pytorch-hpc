#!/usr/bin/env python3
"""
FSDP Training Example with ResNet on CIFAR-10

Demonstrates Fully Sharded Data Parallel (FSDP) training using PyTorch's
native FSDP implementation (torch.distributed.fsdp).

FSDP shards model parameters, gradients, and optimizer states across GPUs,
enabling training of models too large for a single GPU's memory.

Usage:
    # Single node, 4 GPUs:
    mpiexec -n 4 --ppn 4 --cpu-bind none python resnet_fsdp_training.py
    torchrun --standalone --nproc_per_node=4 resnet_fsdp_training.py

    # Multi-node (2 nodes, 4 GPUs each):
    mpiexec -n 8 --ppn 4 --cpu-bind none python resnet_fsdp_training.py
"""

import os
import sys
import time
import argparse
import functools

# Add repo root to path so `from utils...` works
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

import torchvision
import torchvision.transforms as transforms

import torch.distributed as dist
from utils.distributed import init_distributed, cleanup_distributed

def get_cifar10_loaders(batch_size, world_size, rank):
    """Create CIFAR-10 train and test data loaders with DistributedSampler."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # Rank 0 downloads first, then barrier ensures all ranks wait
    if rank == 0:
        torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
        torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    if world_size > 1:
        dist.barrier()

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False,
        transform=transform_train,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False,
        transform=transform_test,
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, test_loader, train_sampler


class Trainer:
    """Handles FSDP training lifecycle."""

    def __init__(self, model, train_loader, test_loader, train_sampler,
                 optimizer, local_rank, world_rank):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_sampler = train_sampler
        self.optimizer = optimizer
        self.local_rank = local_rank
        self.world_rank = world_rank
        self.device = torch.device(f"cuda:{local_rank}")

    def train_epoch(self, epoch):
        self.model.train()
        self.train_sampler.set_epoch(epoch)

        total_loss = 0.0
        correct = 0
        total = 0

        start = time.time()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        elapsed = time.time() - start
        acc = 100.0 * correct / total

        if self.world_rank == 0:
            print(f"  Epoch {epoch:3d}  "
                  f"loss={total_loss / len(self.train_loader):.4f}  "
                  f"acc={acc:.1f}%  "
                  f"time={elapsed:.1f}s")

    def test(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        acc = 100.0 * correct / total
        if self.world_rank == 0:
            print(f"  Test accuracy: {acc:.1f}%")
        return acc


def main():
    parser = argparse.ArgumentParser(description="FSDP ResNet Training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--use-amp", action="store_true",
                        help="Enable BFloat16 mixed precision")
    args = parser.parse_args()

    # Initialize distributed (handles all launchers automatically)
    world_rank, world_size, local_rank = init_distributed()

    if world_rank == 0:
        print(f"{'=' * 60}")
        print(f"FSDP Training with ResNet-18 on CIFAR-10")
        print(f"{'=' * 60}")
        print(f"  World size:    {world_size}")
        print(f"  Batch size:    {args.batch_size} (per GPU)")
        print(f"  Epochs:        {args.epochs}")
        print(f"  Mixed prec:    {'BFloat16' if args.use_amp else 'Float32'}")
        print(f"{'=' * 60}\n")

    # Data
    train_loader, test_loader, train_sampler = get_cifar10_loaders(
        args.batch_size, world_size, world_rank
    )

    # Model
    model = torchvision.models.resnet18(num_classes=10)
    # Adjust first conv for CIFAR-10 (32x32 images instead of 224x224)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)
    model.maxpool = nn.Identity()
    model = model.to(local_rank)

    # FSDP wrapping with auto wrap policy
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1e5
    )

    mixed_precision = None
    if args.use_amp:
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_id=local_rank,
    )

    if world_rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {param_count:,}")
        print(f"  Sharding: FULL_SHARD\n")

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )

    # Train
    trainer = Trainer(
        model, train_loader, test_loader, train_sampler,
        optimizer, local_rank, world_rank,
    )

    for epoch in range(1, args.epochs + 1):
        trainer.train_epoch(epoch)

    trainer.test()

    # Cleanup
    cleanup_distributed()

    if world_rank == 0:
        print(f"\n{'=' * 60}")
        print("FSDP training complete!")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
