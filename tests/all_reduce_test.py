# Description: This script is used to benchmark the performance of the all_reduce and broadcast operations using the nccl and gloo backends.

import os
import time
import argparse
import statistics

import torch
import torch.distributed as dist

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])

if WORLD_RANK == 0:
    print('----------------------')
    print('LOCAL_RANK  : ', LOCAL_RANK)
    print('WORLD_SIZE  : ', WORLD_SIZE)
    print('WORLD_RANK  : ', WORLD_RANK)
    print('cuda device : ', torch.cuda.device_count())
    print('pytorch version : ', torch.__version__)
    print('nccl version : ', torch.cuda.nccl.version())
    print('----------------------')    


def run_broadcast(backend, timing_list):
    tensor = torch.ones((1000, 1000)) 
    # Need to put tensor on a GPU device for nccl backend
    if backend == 'nccl':
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        tensor = tensor.to(device)
    elif backend == 'gloo':
        device = torch.device('cpu')
        tensor = tensor.to(device)

    torch.cuda.synchronize()
    start_time = time.time()

    dist.broadcast(tensor, src=0)

    torch.cuda.synchronize()  # Ensure all operations completed
    end_time = time.time()

    if WORLD_RANK == 0:
        total_time = end_time - start_time
        print(f"{backend}: broadcast {total_time} sec")
        timing_list.append(total_time)


def run_all_reduce(backend, timing_list):
    tensor = torch.ones((1000, 1000)) 
    # Need to put tensor on a GPU device for nccl backend
    if backend == 'nccl':
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        tensor = tensor.to(device)
    elif backend == 'gloo':
        device = torch.device('cpu')
        tensor = tensor.to(device)

    torch.cuda.synchronize()
    start_time = time.time()

    dist.all_reduce(tensor)

    torch.cuda.synchronize()  # Ensure all operations completed
    end_time = time.time()

    if WORLD_RANK == 0:
        total_time = end_time - start_time
        print(f"{backend}: all_reduce {total_time} sec")
        timing_list.append(total_time)


def init_processes(backend):
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)

    # Warmup runs
    warmup_runs = 2
    warmup_time_broadcast = []
    warmup_time_all_reduce = []

    for _ in range(warmup_runs):
        run_broadcast(backend, warmup_time_broadcast)
        run_all_reduce(backend, warmup_time_all_reduce)

    # Benchmark runs
    benchmark_runs = 20
    benchmark_time_broadcast = []
    benchmark_time_all_reduce = []

    for _ in range(benchmark_runs):
        run_broadcast(backend, benchmark_time_broadcast)
        run_all_reduce(backend, benchmark_time_all_reduce)

    if WORLD_RANK == 0:
        warmup_broadcast = statistics.mean(warmup_time_broadcast)
        benchmark_broadcast = statistics.mean(benchmark_time_broadcast)
        warmup_all_reduce = statistics.mean(warmup_time_all_reduce)
        benchmark_all_reduce = statistics.mean(benchmark_time_all_reduce)

        print(f"{backend}: broadcast warmup: {warmup_broadcast} sec, benchmark time: {benchmark_broadcast} sec")
        print(f"{backend}: all_reduce warmup: {warmup_all_reduce} sec, benchmark time: {benchmark_all_reduce} sec")
        
        log_file_path = "benchmark_results.log"
        with open(log_file_path, 'a') as log_file:
            if backend == 'nccl':
                nccl_version= '-'.join(map(str, torch.cuda.nccl.version()))
                log_file.write(f"{backend} {nccl_version}: broadcast warmup: {warmup_broadcast} sec, benchmark time: {benchmark_broadcast} sec\n")
                log_file.write(f"{backend} {nccl_version}: all_reduce warmup: {warmup_all_reduce} sec, benchmark time: {benchmark_all_reduce} sec\n")
            else:
                log_file.write(f"{backend} \t   : broadcast warmup: {warmup_broadcast} sec, benchmark time: {benchmark_broadcast} sec\n")
                log_file.write(f"{backend} \t   : all_reduce warmup: {warmup_all_reduce} sec, benchmark time: {benchmark_all_reduce} sec\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--backend", type=str, default="nccl", choices=['nccl', 'gloo'])
    args = parser.parse_args()

    init_processes(backend=args.backend)