import os
import time
import argparse
import statistics
import socket
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

def run_broadcast(backend, tensor, timing_list):
    if backend == "nccl":
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        tensor = tensor.to(device)
    elif backend == "gloo":
        device = torch.device("cpu")
        tensor = tensor.to(device)

    torch.cuda.synchronize()
    start_time = time.time()

    dist.broadcast(tensor, src=0)

    torch.cuda.synchronize()
    end_time = time.time()

    if WORLD_RANK == 0:
        total_time = end_time - start_time
        timing_list.append(total_time)

def run_all_reduce(backend, tensor, timing_list):
    if backend == "nccl":
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        tensor = tensor.to(device)
    elif backend == "gloo":
        device = torch.device("cpu")
        tensor = tensor.to(device)

    torch.cuda.synchronize()
    start_time = time.time()

    dist.all_reduce(tensor)

    torch.cuda.synchronize()
    end_time = time.time()

    if WORLD_RANK == 0:
        total_time = end_time - start_time
        timing_list.append(total_time)

def init_processes(backend, tensor_sizes):
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)

    for size in tensor_sizes:
        tensor = torch.ones(size)
        tensor_size_in_bytes = tensor.element_size() * tensor.numel()

        if WORLD_RANK == 0:
            print(f"Running benchmarks for tensor size: {size}, Memory: {tensor_size_in_bytes / (1024**2):.2f} MB")

        warmup_runs = 2
        warmup_time_broadcast = []
        warmup_time_all_reduce = []

        for _ in range(warmup_runs):
            run_broadcast(backend, tensor, warmup_time_broadcast)
            run_all_reduce(backend, tensor, warmup_time_all_reduce)

        benchmark_runs = 20
        benchmark_time_broadcast = []
        benchmark_time_all_reduce = []

        for _ in range(benchmark_runs):
            run_broadcast(backend, tensor, benchmark_time_broadcast)
            run_all_reduce(backend, tensor, benchmark_time_all_reduce)

        if WORLD_RANK == 0:
            warmup_broadcast = statistics.mean(warmup_time_broadcast)
            benchmark_broadcast = statistics.mean(benchmark_time_broadcast)
            warmup_all_reduce = statistics.mean(warmup_time_all_reduce)
            benchmark_all_reduce = statistics.mean(benchmark_time_all_reduce)

            print(
                f"{backend} (size {size:.1e}, {tensor_size_in_bytes / (1024**2):.3f} MB): "
                f"broadcast warmup: {warmup_broadcast} sec, benchmark time: {benchmark_broadcast} sec"
            )
            print(
                f"{backend} (size {size:.1e}, {tensor_size_in_bytes / (1024**2):.3f} MB): "
                f"all_reduce warmup: {warmup_all_reduce} sec, benchmark time: {benchmark_all_reduce} sec"
            )

            log_file_path = "benchmark_results.log"
            with open(log_file_path, "a") as log_file:
                if backend == "nccl":
                    nccl_version = "-".join(map(str, torch.cuda.nccl.version()))
                    log_file.write(
                        f"{backend} {nccl_version} (size {size:.1e}, {tensor_size_in_bytes / (1024**2):.3f} MB): "
                        f"broadcast warmup: {warmup_broadcast} sec, benchmark time: {benchmark_broadcast} sec\n"
                    )
                    log_file.write(
                        f"{backend} {nccl_version} (size {size:.1e} {tensor_size_in_bytes / (1024**2):.3f} MB): "
                        f"all_reduce warmup: {warmup_all_reduce} sec, benchmark time: {benchmark_all_reduce} sec\n"
                    )
                else:
                    log_file.write(
                        f"{backend} \t   (size {size:.1e}, {tensor_size_in_bytes / (1024**2):.3f} MB): "
                        f"broadcast warmup: {warmup_broadcast} sec, benchmark time: {benchmark_broadcast} sec\n"
                    )
                    log_file.write(
                        f"{backend} \t   (size {size:.1e}, {tensor_size_in_bytes / (1024**2):.3f} MB): "
                        f"all_reduce warmup: {warmup_all_reduce} sec, benchmark time: {benchmark_all_reduce} sec\n"
                    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank",
        type=int,
        help="Local rank. Necessary for using the torch.distributed.launch utility.",
    )
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo","mpi"])
    parser.add_argument("--tensor_sizes", nargs="+", type=int, default=[(1000, 1000), (10000, 10000), (100000, 100000)],
                        help="List of tensor sizes to benchmark. Example: --tensor_sizes 10000 1000 1000 5000")
    args = parser.parse_args()

    tensor_sizes = [(s,) if isinstance(s, int) else s for s in args.tensor_sizes]
    tensor_sizes = [int(10**i) for i in range(3, 9)]

    print (tensor_sizes)
    init_processes(backend=args.backend, tensor_sizes=tensor_sizes)
