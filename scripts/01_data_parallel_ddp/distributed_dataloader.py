#!/usr/bin/env python
"""
Example code demonstrating DistributedSampler and DataLoader usage.

Usage:
    mpiexec -n 4 --ppn 4 --cpu-bind none python distributed_dataloader.py
    torchrun --standalone --nproc_per_node=4 distributed_dataloader.py
"""

import os
import sys
import argparse

# Add repo root to path so `from utils...` works
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.distributed import init_distributed, cleanup_distributed


# Custom Dataset
class SimpleDataset(Dataset):
    """Example dataset with synthetic data"""
    def __init__(self, size=1000, input_dim=10):
        self.size = size
        self.input_dim = input_dim
        # Generate random data
        self.data = torch.randn(size, input_dim)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Simple Model
class SimpleModel(nn.Module):
    """Simple neural network model"""
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size per process")
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo", "mpi"])
    parser.add_argument("--dataset_size", type=int, default=1000, help="Size of the dataset")
    parser.add_argument("--input_dim", type=int, default=10, help="Input dimension")

    argv = parser.parse_args()

    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    dataset_size = argv.dataset_size
    input_dim = argv.input_dim

    # Initialize distributed (handles all launchers automatically)
    WORLD_RANK, WORLD_SIZE, LOCAL_RANK = init_distributed(backend=argv.backend)

    device = torch.device(f"cuda:{LOCAL_RANK}")
    print(f"device: {device}, world_rank: {WORLD_RANK}, local_rank: {LOCAL_RANK}")

    # Create dataset
    dataset = SimpleDataset(size=dataset_size, input_dim=input_dim)

    # Create DistributedSampler - same pattern as scripts/main.py line 277
    # This restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=dataset)

    # Create DataLoader with DistributedSampler - same pattern as scripts/main.py lines 279-284
    # Important: Do NOT use shuffle=True when using DistributedSampler
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
    )

    if WORLD_RANK == 0:
        print(f"Total dataset size: {len(dataset)}")
        print(f"Number of batches per GPU: {len(train_loader)}")
        print(f"Samples per GPU: ~{len(dataset) // WORLD_SIZE}")

    # Create model
    model = SimpleModel(input_dim=input_dim)
    model = model.to(LOCAL_RANK)

    # Wrap with DistributedDataParallel - same as scripts/main.py lines 240-242
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[LOCAL_RANK],
        output_device=LOCAL_RANK
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Local Rank: {LOCAL_RANK}, GPU: {WORLD_RANK}, Epoch: {epoch}, Training ...")

        # IMPORTANT: Set epoch for sampler - same as scripts/main.py line 334
        # This ensures different shuffling each epoch
        train_loader.sampler.set_epoch(epoch)

        ddp_model.train()

        epoch_loss = 0.0
        for batch_idx, (data, labels) in enumerate(train_loader):
            # Move data to GPU
            data = data.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = ddp_model(data)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Print progress
            if batch_idx % 10 == 0:
                print(f"[Rank {WORLD_RANK}] Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Synchronize all processes
        torch.cuda.synchronize()

        # Print epoch summary from rank 0
        if WORLD_RANK == 0:
            avg_loss = epoch_loss / len(train_loader)
            print("-" * 75)
            print(f"Epoch: {epoch}, Average Loss: {avg_loss:.4f}")
            print("-" * 75)

    if WORLD_RANK == 0:
        print("Training completed successfully!")

    cleanup_distributed()


if __name__ == "__main__":
    main()
