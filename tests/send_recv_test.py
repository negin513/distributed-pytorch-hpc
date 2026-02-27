"""
Send/recv communication test between ranks.

mpiexec -n 8 --ppn 4 --cpu-bind none python send_recv_test.py
mpiexec -n 4 --ppn 4 --cpu-bind none python send_recv_test.py
torchrun --standalone --nproc_per_node=4 send_recv_test.py
"""
import os
import sys
import time
import argparse
import statistics

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
    # Process group already initialized by init_distributed() above
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
