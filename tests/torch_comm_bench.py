import os
import sys
import time
import socket
import argparse
import statistics

# Add repo root to path so `from utils...` works
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.distributed as dist

from utils.distributed import init_distributed, cleanup_distributed

WORLD_RANK, WORLD_SIZE, LOCAL_RANK = init_distributed(verbose=False)


if WORLD_RANK == 0:
    print("=" * 60)
    print("DISTRIBUTED COMMUNICATION BENCHMARK")
    print(f"World Size       : {WORLD_SIZE}")
    print(f"CUDA Devices     : {torch.cuda.device_count()}")
    print(f"PyTorch Version  : {torch.__version__}")
    print(f"CUDA Version     : {torch.version.cuda}")
    print(f"NCCL Version     : {torch.cuda.nccl.version()}")
    print("Torch Config     : ")
    print(torch.__config__.show())
    print(torch.__config__.parallel_info())
    print("=" * 60)

def run_broadcast(backend, tensor, timing_list):
    if backend == "nccl":
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        tensor = tensor.to(device)
    elif backend == "gloo":
        device = torch.device("cpu")
        tensor = tensor.to(device)

    # Better synchronization
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if WORLD_SIZE > 1:
        dist.barrier()
    
    start_time = time.perf_counter()
    
    if WORLD_SIZE > 1:
        dist.broadcast(tensor, src=0)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if WORLD_SIZE > 1:
        dist.barrier()
    
    end_time = time.perf_counter()

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

    # Better synchronization
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if WORLD_SIZE > 1:
        dist.barrier()
    
    start_time = time.perf_counter()
    
    if WORLD_SIZE > 1:
        dist.all_reduce(tensor)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if WORLD_SIZE > 1:
        dist.barrier()
    
    end_time = time.perf_counter()

    if WORLD_RANK == 0:
        total_time = end_time - start_time
        timing_list.append(total_time)

def calculate_bandwidth(tensor_size_bytes, time_seconds, operation, world_size):
    """
    Calculate bandwidth for communication operations.
    
    Bandwidth metrics:
    - Broadcast: Simple bandwidth (data_size / time)
      Data is sent once from root to all other ranks.
    
    - All-reduce: Bus bandwidth using ring algorithm formula
      In a ring all-reduce, each GPU sends and receives (n-1)/n of the data
      in both reduce-scatter and all-gather phases, hence 2 * (n-1)/n * size.
      This "bus bandwidth" metric allows comparing against hardware limits.
    """
    if operation == "broadcast":
        # Simple bandwidth: total data transferred / time
        effective_bytes = tensor_size_bytes
    elif operation == "all_reduce":
        # Bus bandwidth formula for ring all-reduce: 2 * (n-1)/n * size
        # This accounts for both reduce-scatter and all-gather phases
        effective_bytes = 2 * (world_size - 1) / world_size * tensor_size_bytes
    else:
        effective_bytes = tensor_size_bytes
    
    bandwidth_gbps = (effective_bytes / (1024**3)) / time_seconds
    return bandwidth_gbps

def log_results(log_file_path, backend, operation, tensor_size_elements, tensor_size_bytes, 
                warmup_times, benchmark_times, warmup_runs, benchmark_runs):
    """Log detailed results to file in comprehensive CSV format."""
    if not benchmark_times:
        return
    
    # Calculate statistics
    warmup_mean = statistics.mean(warmup_times) if warmup_times else 0
    warmup_std = statistics.stdev(warmup_times) if len(warmup_times) > 1 else 0
    benchmark_mean = statistics.mean(benchmark_times)
    benchmark_std = statistics.stdev(benchmark_times) if len(benchmark_times) > 1 else 0
    benchmark_min = min(benchmark_times)
    benchmark_max = max(benchmark_times)
    
    # Calculate bandwidth
    bandwidth = calculate_bandwidth(tensor_size_bytes, benchmark_mean, operation, WORLD_SIZE)
    
    tensor_size_mb = tensor_size_bytes / (1024**2)
    
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{backend},{operation},{tensor_size_elements},{tensor_size_bytes},{tensor_size_mb:.6f},")
        log_file.write(f"{warmup_runs},{benchmark_runs},")
        log_file.write(f"{warmup_mean:.8f},{warmup_std:.8f},")
        log_file.write(f"{benchmark_mean:.8f},{benchmark_std:.8f},{benchmark_min:.8f},{benchmark_max:.8f},")
        log_file.write(f"{bandwidth:.6f}\n")

def init_processes(backend, tensor_sizes, warmup_runs, benchmark_runs, log_file_path):
    # Process group already initialized by init_distributed() above

    if WORLD_RANK == 0:
        print(f"Initialized {backend.upper()} backend")

        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    if WORLD_RANK == 0:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        hostname = socket.gethostname()
        
        # Get conda environment
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'N/A')
        
        gpu_info = "N/A"
        gpu_memory = "N/A"
        if torch.cuda.is_available() and cuda_devices > 0:
            gpu_info = torch.cuda.get_device_name(0)
            gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        
        nccl_version = "N/A"
        if hasattr(torch.cuda, 'nccl'):
            nccl_version = ".".join(map(str, torch.cuda.nccl.version()))
        
        with open(log_file_path, "a") as f:
            f.write(f"# ============================================================\n")
            f.write(f"# Timestamp             : {timestamp}\n")
            f.write(f"# Hostname              : {hostname}\n")
            f.write(f"# World Size            : {WORLD_SIZE}\n")
            f.write(f"# Backend               : {backend}\n")
            if backend == "nccl":
                f.write(f"# NCCL Version          : {nccl_version}\n")
            f.write(f"# Conda Environment     : {conda_env}\n")
            f.write(f"# PyTorch Version       : {torch.__version__}\n")
            f.write(f"# CUDA Version          : {torch.version.cuda}\n")
            f.write(f"# CUDA Devices          : {cuda_devices}\n")
            f.write(f"# GPU Model             : {gpu_info}\n")
            f.write(f"# GPU Memory per Device : {gpu_memory}\n")
            f.write(f"# Warmup Iterations     : {warmup_runs}\n")
            f.write(f"# Benchmark Iterations  : {benchmark_runs}\n")
            f.write(f"# Tensor Sizes          : {tensor_sizes}\n")
            f.write(f"# ------------------------------------------------------------\n")
            f.write(f"backend,operation,tensor_elements,tensor_bytes,tensor_mb,")
            f.write(f"warmup_iterations,benchmark_iterations,")
            f.write(f"warmup_mean_sec,warmup_std_sec,")
            f.write(f"benchmark_mean_sec,benchmark_std_sec,benchmark_min_sec,benchmark_max_sec,")
            f.write(f"bandwidth_gbps\n")

    for idx, size in enumerate(tensor_sizes):
        tensor = torch.ones(size)
        tensor_size_bytes = tensor.element_size() * tensor.numel()

        if WORLD_RANK == 0:
            print(f"\nBenchmark {idx + 1}/{len(tensor_sizes)}")
            print(f"Tensor size: {size:,} elements ({tensor_size_bytes / (1024**2):.2f} MB)")
            print(f"Running {warmup_runs} warmup + {benchmark_runs} benchmark iterations")

        warmup_time_broadcast = []
        warmup_time_all_reduce = []

        for _ in range(warmup_runs):
            run_broadcast(backend, tensor.clone(), warmup_time_broadcast)
            run_all_reduce(backend, tensor.clone(), warmup_time_all_reduce)

        benchmark_time_broadcast = []
        benchmark_time_all_reduce = []

        for _ in range(benchmark_runs):
            run_broadcast(backend, tensor.clone(), benchmark_time_broadcast)
            run_all_reduce(backend, tensor.clone(), benchmark_time_all_reduce)
            
        if WORLD_RANK == 0:
            if warmup_time_broadcast and benchmark_time_broadcast:
                warmup_bc = statistics.mean(warmup_time_broadcast)
                benchmark_bc = statistics.mean(benchmark_time_broadcast)
                bc_std = statistics.stdev(benchmark_time_broadcast) if len(benchmark_time_broadcast) > 1 else 0
                bc_bw = calculate_bandwidth(tensor_size_bytes, benchmark_bc, "broadcast", WORLD_SIZE)
                
                print(f"Broadcast    - Warmup: {warmup_bc:.6f}s, Benchmark: {benchmark_bc:.6f}±{bc_std:.6f}s, BW: {bc_bw:.2f} GB/s")

            if warmup_time_all_reduce and benchmark_time_all_reduce:
                warmup_ar = statistics.mean(warmup_time_all_reduce)
                benchmark_ar = statistics.mean(benchmark_time_all_reduce)
                ar_std = statistics.stdev(benchmark_time_all_reduce) if len(benchmark_time_all_reduce) > 1 else 0
                ar_bw = calculate_bandwidth(tensor_size_bytes, benchmark_ar, "all_reduce", WORLD_SIZE)
                
                print(f"All-reduce   - Warmup: {warmup_ar:.6f}s, Benchmark: {benchmark_ar:.6f}±{ar_std:.6f}s, BW: {ar_bw:.2f} GB/s")

            # Log to file
            log_results(log_file_path, backend+"-"+nccl_version, "broadcast", size, tensor_size_bytes,
                       warmup_time_broadcast, benchmark_time_broadcast, warmup_runs, benchmark_runs)
            log_results(log_file_path, backend+"-"+nccl_version, "all_reduce", size, tensor_size_bytes,
                       warmup_time_all_reduce, benchmark_time_all_reduce, warmup_runs, benchmark_runs)


    if WORLD_RANK == 0:
        print(f"\nBenchmark completed!")
        print(f"Results saved to: {log_file_path}")

    if dist.is_initialized():
        dist.barrier()
        time.sleep(1)
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed Communication Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo", "mpi"],
                       help='Distributed backend to use ("nccl" for GPU, "gloo" for CPU, "mpi" for MPI backend)')
    parser.add_argument("--tensor_sizes", nargs="+", type=int,
                    default=[10**3,10**4,10**5,10**6,10**7,10**8,],)
    parser.add_argument("--warmup_runs", type=int, default=5,
                       help="Number of warmup iterations")
    parser.add_argument("--benchmark_runs", type=int, default=20,
                       help="Number of benchmark iterations")
    parser.add_argument("--output_file", type=str, default="results/benchmark_results.log",
                       help="Output log file (will be saved in results directory)")
    
    args = parser.parse_args()

    tensor_sizes = args.tensor_sizes

    if WORLD_RANK == 0:
        print(f"Tensor sizes: {[f'{s:,}' for s in tensor_sizes]}")
        print(f"Backend: {args.backend}")
        print(f"Warmup runs: {args.warmup_runs}")
        print(f"Benchmark runs: {args.benchmark_runs}")

    init_processes(backend=args.backend, tensor_sizes=tensor_sizes, 
                  warmup_runs=args.warmup_runs, benchmark_runs=args.benchmark_runs,
                  log_file_path=args.output_file)
