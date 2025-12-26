"""
This script initializes the distributed process group and runs a simple 
distributed send/receive operation.
"""

import os
import argparse
import socket 
import torch
import torch.distributed as dist

try: 
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    shmem_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    
    LOCAL_RANK = shmem_comm.Get_rank()
    WORLD_SIZE = comm.Get_size()
    WORLD_RANK = comm.Get_rank()

    os.environ['MASTER_ADDR'] = comm.bcast( socket.gethostbyname( socket.gethostname() ), root=0 )
    os.environ['MASTER_PORT'] =	'1234'

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

print (LOCAL_RANK, WORLD_SIZE, WORLD_RANK) 

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


def init_processes(backend):
    """
    Initialize the distributed environment for PyTorch and call run function. 
    """
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    run(backend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank",
        type=int,
        help="Local rank. Necessary for using the torch.distributed.launch utility.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"])
    args = parser.parse_args()

    init_processes(backend=args.backend)
