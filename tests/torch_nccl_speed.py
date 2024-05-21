# 05-02-2024.

# this script include the code to test the communication between different ranks

import os
import time

import torch
import torch.distributed as dist

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])

print ('----------------------')
print ('local rank  : ', LOCAL_RANK)
print ('world size  : ', WORLD_SIZE)
print ('world rank  : ', WORLD_RANK)
print ('cuda device : ', torch.cuda.device_count())
print ('pytorch version : ', torch.__version__)
print ('----------------------')    


def run(backend):
    
    tensor = torch.ones(100000)
    device = torch.device("cuda:{}".format(LOCAL_RANK))
    tensor = tensor.to(device)

    torch.cuda.synchronize()
    start_time = time.time()

    # Warm-up
    for _ in range(10):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # Synchronize all processes before starting timing
    dist.barrier()
    start_time = time.time()
    
    # Perform 100 all-reduce operations
    for _ in range(100):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # Synchronize all processes and measure the time taken
    dist.barrier()
    total_time = time.time() - start_time
    
    if LOCAL_RANK == 0:
        print(f"Total time for 100 all-reduce operations: {total_time:.4f} seconds")
        print(f"Average time per all-reduce: {total_time / 100:.4f} seconds")


    torch.cuda.synchronize()  # Ensure all operations completed
    end_time = time.time()

    # Calculating the duration of the send/receive operation
    print(f'Rank {WORLD_RANK}: Operation took {end_time - start_time:.6f} seconds')


def init_processes(backend):
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    run(backend)

if __name__ == "__main__":
    backend = "nccl"
    init_processes(backend)


