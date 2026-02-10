#!/usr/bin/env python
"""
Example code demonstrating DistributedSampler and DataLoader usage
with MPI setup from scripts/main.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import os
import socket
import numpy as np
import argparse


# MPI Setup - same as scripts/main.py
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    shmem_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

    LOCAL_RANK = shmem_comm.Get_rank()
    WORLD_SIZE = comm.Get_size()
    WORLD_RANK = comm.Get_rank()

except:
    if "LOCAL_RANK" in os.environ:
        # Environment variables set by torch.distributed.launch or torchrun
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        WORLD_RANK = int(os.environ["RANK"])
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        # Environment variables set by mpirun
        LOCAL_RANK = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        WORLD_SIZE = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        WORLD_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])
    elif "PMI_RANK" in os.environ:
        # Environment variables set by cray-mpich
        LOCAL_RANK = int(os.environ["PMI_LOCAL_RANK"])
        WORLD_SIZE = int(os.environ["PMI_SIZE"])
        WORLD_RANK = int(os.environ["PMI_RANK"])
    else:
        import sys
        sys.exit("Can't find the evironment variables for local rank")

if "MASTER_ADDR" not in os.environ:
    os.environ['MASTER_ADDR'] = comm.bcast(socket.gethostbyname(socket.gethostname()), root=0)
if "MASTER_PORT" not in os.environ:
    os.environ['MASTER_PORT'] = str(np.random.randint(1000, 8000))


if WORLD_RANK == 0:
    print('----------------------')
    print('LOCAL_RANK  : ', LOCAL_RANK)
    print('WORLD_SIZE  : ', WORLD_SIZE)
    print('WORLD_RANK  : ', WORLD_RANK)
    print("cuda device : ", torch.cuda.device_count())
    print("pytorch version : ", torch.__version__)
    print('----------------------')


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
    parser.add_argument("--local_rank", type=int, help="Local rank")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size per process")
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo", "mpi"])
    parser.add_argument("--dataset_size", type=int, default=1000, help="Size of the dataset")
    parser.add_argument("--input_dim", type=int, default=10, help="Input dimension")

    argv = parser.parse_args()

    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    backend = argv.backend
    dataset_size = argv.dataset_size
    input_dim = argv.input_dim

    # Initialize distributed backend - same as scripts/main.py
    torch.distributed.init_process_group(
        backend=backend,
        rank=WORLD_RANK,
        world_size=WORLD_SIZE
    )
    torch.cuda.set_device(LOCAL_RANK)

    device = torch.device("cuda:{}".format(LOCAL_RANK))
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


if __name__ == "__main__":
    main()
