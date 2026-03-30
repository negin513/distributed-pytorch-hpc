"""
Shared distributed initialization utilities.

Supports multiple launchers commonly used on NCAR's Derecho:
    - torchrun (torch.distributed.launch)
    - OpenMPI (mpirun / mpiexec with OMPI env vars)
    - Cray MPICH (mpiexec with PMI env vars)
    - mpi4py fallback
    - Single-process (no distributed)

Usage:
    from utils.distributed import init_distributed, cleanup_distributed, is_main_rank

    rank, world_size, local_rank = init_distributed()
    # ... training code ...
    cleanup_distributed()
"""

import os
import socket

import torch
import torch.distributed as dist


def get_rank_info():
    """
    Detect LOCAL_RANK, WORLD_SIZE, WORLD_RANK from the active launcher.

    Checks environment variables in priority order:
        1. torchrun (LOCAL_RANK, RANK, WORLD_SIZE)
        2. OpenMPI  (OMPI_COMM_WORLD_*)
        3. Cray MPICH (PMI_RANK, PMI_SIZE, PMI_LOCAL_RANK)
        4. mpi4py   (fallback for mpiexec without env vars)
        5. Single process

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
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        _set_master_addr_from_mpi(world_rank)
        return local_rank, world_size, world_rank, "openmpi"

    # Method 3: Cray MPICH (PMI)
    if "PMI_RANK" in os.environ:
        world_size = int(os.environ["PMI_SIZE"])
        world_rank = int(os.environ["PMI_RANK"])

        # Determine local rank: try PMI_LOCAL_RANK, then PALS_LOCAL_RANKID
        # (set by Derecho's PALS scheduler), then compute from GPU count.
        if "PMI_LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["PMI_LOCAL_RANK"])
        elif "PALS_LOCAL_RANKID" in os.environ:
            local_rank = int(os.environ["PALS_LOCAL_RANKID"])
        else:
            gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
            local_rank = world_rank % gpus_per_node

        _set_master_addr_from_mpi(world_rank)
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
    except Exception:
        pass

    # Method 5: Single process (no distributed)
    return 0, 1, 0, "single"


def _set_master_addr_from_mpi(world_rank):
    """Set MASTER_ADDR and MASTER_PORT for MPI-based launchers."""
    if "MASTER_ADDR" not in os.environ:
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            if world_rank == 0:
                master_addr = socket.gethostbyname(socket.gethostname())
            else:
                master_addr = None
            master_addr = comm.bcast(master_addr, root=0)
            os.environ["MASTER_ADDR"] = master_addr
        except ImportError:
            # Fallback: use current hostname (works for single-node)
            os.environ["MASTER_ADDR"] = socket.gethostbyname(
                socket.gethostname()
            )
    os.environ.setdefault("MASTER_PORT", "29500")


def init_distributed(backend="nccl", verbose=True):
    """
    Initialize the distributed process group and set the CUDA device.

    Detects the launcher automatically, sets MASTER_ADDR/PORT if needed,
    calls ``dist.init_process_group``, and assigns the correct GPU.

    Args:
        backend: Communication backend ("nccl", "gloo", "mpi").
        verbose: If True, rank 0 prints configuration info.

    Returns:
        tuple: (world_rank, world_size, local_rank)
    """
    local_rank, world_size, world_rank, launcher = get_rank_info()

    if world_size > 1 and not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=world_rank,
        )

    elif world_size == 1 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if verbose and world_rank == 0:
        print(f"  Distributed init: launcher={launcher}, "
              f"world_size={world_size}, backend={backend}")

    return world_rank, world_size, local_rank


def cleanup_distributed():
    """Destroy the distributed process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_rank():
    """Return True if this is rank 0 (or if not distributed)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_rank():
        print(*args, **kwargs)
