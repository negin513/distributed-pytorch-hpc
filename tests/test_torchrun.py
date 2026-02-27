"""
Simple distributed send/receive test.

mpiexec -n 4 --ppn 4 --cpu-bind none python test_torchrun.py
torchrun --standalone --nproc_per_node=4 test_torchrun.py
"""

import os
import sys
import argparse

# Add repo root to path so `from utils...` works
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.distributed as dist

from utils.distributed import init_distributed, cleanup_distributed

WORLD_RANK, WORLD_SIZE, LOCAL_RANK = init_distributed(verbose=False)
print(LOCAL_RANK, WORLD_SIZE, WORLD_RANK)

if WORLD_RANK == 0:
    print("----------------------")
    print("WORLD_SIZE  : ", WORLD_SIZE)
    print("cuda device : ", torch.cuda.device_count())
    print("pytorch version : ", torch.__version__)
    print("nccl version : ", torch.cuda.nccl.version())
    print("torch config : ", torch.__config__.show())
    print(torch.__config__.parallel_info())
    print("----------------------")


def run(backend):
    """
    The function initializes a tensor, moves it to the appropriate device,
    and then sends or receives the tensor based on the process's world rank.

    Args:
        backend (str): The backend to use for distributed processing ('nccl' or 'gloo').

    """
    tensor = torch.zeros(1)

    # Need to put tensor on a GPU device for nccl backend
    if backend == "nccl":
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        tensor = tensor.to(device)
    elif backend == "gloo":
        device = torch.device("cpu")  # Gloo backend works with CPU
        tensor = tensor.to(device)
    

    if WORLD_RANK == 0:
        for rank_recv in range(1, WORLD_SIZE):
            dist.send(tensor=tensor, dst=rank_recv)
            print("worker_{} sent data to Rank {}\n".format(0, rank_recv))
    else:
        dist.recv(tensor=tensor, src=0)
        print("worker_{} has received data from rank {}\n".format(WORLD_RANK, 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"])
    args = parser.parse_args()

    run(backend=args.backend)
