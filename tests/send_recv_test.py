"""
# created: 09-19-2023 and last updated: 02-05-2024.
# adapted from : https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide#launch-multi-node-pytorch-distributed-applications

# this script include the code to test the communication between different ranks
mpiexec -n 8 --ppn 4 --cpu-bind none python all_reduce_test.py
""" 
import os
import time
import argparse
import statistics

import torch
import torch.distributed as dist
import socket

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


def run(backend, timing_list):
    # tensor = torch.zeros(1000000)
    tensor = torch.zeros((10000, 1000))
    # Need to put tensor on a GPU device for nccl backend
    if backend == "nccl":
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        tensor = tensor.to(device)
    elif backend == "gloo":
        device = torch.device("cpu")
        tensor = tensor.to(device)

    torch.cuda.synchronize()
    start_time = time.time()

    if WORLD_RANK == 0:
        for rank_recv in range(1, WORLD_SIZE):
            dist.send(tensor=tensor, dst=rank_recv)
            # print('worker_{} sent data to Rank {}\n'.format(0, rank_recv))
    else:
        dist.recv(tensor=tensor, src=0)
        # print('worker_{} has received data from rank {}\n'.format(WORLD_RANK, 0))

    torch.cuda.synchronize()  # Ensure all operations completed
    end_time = time.time()

    if WORLD_RANK == 0:
        total_time = end_time - start_time
        print(f"{backend}: {total_time} sec")
        timing_list.append(total_time)
    # Calculating the duration of the send/receive operation
    # print(f'Rank {WORLD_RANK}: Operation took {end_time - start_time:.6f} seconds')


def init_processes(backend):
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)

    warmup_runs = 2
    warmup_time = []
    for _ in range(warmup_runs):
        run(backend, warmup_time)

    benchmark_runs = 20
    benchmark_time = []
    for _ in range(benchmark_runs):
        run(backend, benchmark_time)
    # run(backend)
    # print (warmup_time)
    if WORLD_RANK == 0:
        warmup = statistics.mean(warmup_time)
        benchmark = statistics.mean(benchmark_time)

        print(f"{backend}: warmup: {warmup} sec benchmark time: {benchmark} sec")
        log_file_path = "benchmark_results.log"
        with open(log_file_path, "a") as log_file:
            if backend == "nccl":
                nccl_version = "-".join(map(str, torch.cuda.nccl.version()))
                log_file.write(
                    f"{backend} {nccl_version}: send/recv warmup: {warmup} sec, benchmark time: {benchmark} sec.\n"
                )
            else:
                log_file.write(
                    f"{backend}       : send/recv warmup: {warmup} sec, benchmark time: {benchmark} sec.\n"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank",
        type=int,
        help="Local rank. Necessary for using the torch.distributed.launch utility.",
    )
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    args = parser.parse_args()

    init_processes(backend=args.backend)
