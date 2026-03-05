#!/usr/bin/env python
"""
Example code demonstrating DistributedSampler and DataLoader usage.

Usage:
    # Single GPU (no launcher needed)
    python distributed_dataloader.py

    # Single node, multiple GPUs
    torchrun --standalone --nproc_per_node=4 distributed_dataloader.py

    # Multi-node with MPI (2 nodes, 4 GPUs each)
    mpiexec -n 8 --ppn 4 --cpu-bind none python distributed_dataloader.py
"""
import os
import socket
import argparse


import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group



# =============================================================================
# Distributed Environment Setup
# =============================================================================

def detect_rank_info():
    """
    Detect LOCAL_RANK, WORLD_SIZE, WORLD_RANK from the active launcher.

    Checks environment variables in priority order:
        1. torchrun (LOCAL_RANK, RANK, WORLD_SIZE)
        2. OpenMPI  (OMPI_COMM_WORLD_*)
        3. Cray MPICH (PMI_RANK, PMI_SIZE, PMI_LOCAL_RANK)
        4. mpi4py   (fallback for mpiexec without env vars)
        5. Single process

    Does NOT initialize the process group — use init_distributed() for that.

    Returns:
        tuple: (local_rank, world_size, world_rank, launcher_name)
    """
    # Method 1: torchrun / torch.distributed.launch
    if "LOCAL_RANK" in os.environ and "RANK" in os.environ:
        return (
            int(os.environ["LOCAL_RANK"]),
            int(os.environ["WORLD_SIZE"]),
            int(os.environ["RANK"]),
            "torchrun",
        )

    # Method 2: OpenMPI mpirun
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ.setdefault("MASTER_ADDR", socket.gethostbyname(socket.gethostname()))
        os.environ.setdefault("MASTER_PORT", "29500")
        return (
            int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]),
            int(os.environ["OMPI_COMM_WORLD_SIZE"]),
            int(os.environ["OMPI_COMM_WORLD_RANK"]),
            "openmpi",
        )

    # Method 3: Cray MPICH (PMI)
    if "PMI_RANK" in os.environ:
        world_size = int(os.environ["PMI_SIZE"])
        world_rank = int(os.environ["PMI_RANK"])

        # Determine local rank
        if "PMI_LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["PMI_LOCAL_RANK"])
        elif "PALS_LOCAL_RANKID" in os.environ:
            local_rank = int(os.environ["PALS_LOCAL_RANKID"])
        else:
            gpus_per_node = torch.cuda.device_count() or 1
            local_rank = world_rank % gpus_per_node

        os.environ.setdefault("MASTER_ADDR", socket.gethostbyname(socket.gethostname()))
        os.environ.setdefault("MASTER_PORT", "29500")
        return local_rank, world_size, world_rank, "cray-mpich"

    # Method 4: mpi4py (fallback for mpiexec without env vars)
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()

        if world_size > 1:
            shmem_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
            local_rank = shmem_comm.Get_rank()
            world_rank = comm.Get_rank()

            master_addr = comm.bcast(
                socket.gethostbyname(socket.gethostname()), root=0
            )
            os.environ.setdefault("MASTER_ADDR", master_addr)
            os.environ.setdefault("MASTER_PORT", "29500")

            return local_rank, world_size, world_rank, "mpi4py"
    except ImportError:
        pass

    # Method 5: Single process (no distributed)
    return 0, 1, 0, "single"


def init_distributed(backend="nccl"):
    """
    Initialize distributed training environment.

    Detects the launcher, sets up the device, and initializes the process group.

    Args:
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU)

    Returns:
        tuple: (local_rank, world_size, world_rank, launcher_name)
    """
    local_rank, world_size, world_rank, launcher = detect_rank_info()
    
    # Initialize process group if distributed
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        init_process_group(backend=backend, rank=world_rank, world_size=world_size)

    return local_rank, world_size, world_rank, launcher


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

    # -------------------------------------------
    # 1. Initialize distributed (handles all launchers automatically)
    LOCAL_RANK, WORLD_SIZE, WORLD_RANK, LAUNCHER = init_distributed(backend=argv.backend)

    device = torch.device(f"cuda:{LOCAL_RANK}")

    if WORLD_RANK == 0:
        print(f"Running on {LAUNCHER} with {WORLD_SIZE} processes")
        print("-" * 75)

    # -------------------------------------------
    # 2. Create dataset, sampler, and dataloader
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
        print(f"Total dataset size   : {len(dataset)}")
        print(f"Samples per GPU      : ~{len(dataset) // WORLD_SIZE}")
        print(f"Batches per GPU      : {len(train_loader)}")
        print(f"Effective batch size : {batch_size * WORLD_SIZE}")
        print("-" * 50)

    # -------------------------------------------
    # 3. Create model and move to GPU & wrap with DDP
    model = SimpleModel(input_dim=input_dim)
    model = model.to(LOCAL_RANK)

    # Wrap with DistributedDataParallel - same as scripts/main.py lines 240-242
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[LOCAL_RANK],
        output_device=LOCAL_RANK
    )

    # -------------------------------------------
    # 4. Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01, momentum=0.9)

    # -------------------------------------------
    # 5. Training loop
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

        # Print epoch summary from rank 0
        if WORLD_RANK == 0:
            avg_loss = epoch_loss / len(train_loader)
            print("-" * 75)
            print(f"Epoch: {epoch}, Average Loss: {avg_loss:.4f}")
            print("-" * 75)

        # Optional : Save checkpoint every N epochs -- can be wrapped in a function like _save_snapshot()
        if (epoch + 1) % 1000 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch}.pt"

            if WORLD_RANK == 0:
                torch.save(ddp_model.state_dict(), checkpoint_path)
                print(f"Epoch {epoch} | Model checkpoint saved at {checkpoint_path}")

            if WORLD_SIZE > 1:
                dist.barrier()  # Ensure all processes have saved before next epoch


    # -------------------------------------------
    # Last step: Clean up distributed environment
    if WORLD_SIZE > 1:
        destroy_process_group()

    if WORLD_RANK == 0:
        print("Training completed successfully!")


if __name__ == "__main__":
    main()
