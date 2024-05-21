# created: 09-19-2023 and last updated: 02-05-2024.
# adapted from : https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide#launch-multi-node-pytorch-distributed-applications

# this script include the code to test the communication between different ranks

import os
import time
import argparse

import torch
import torch.distributed as dist

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])

print ('----------------------')
print ('LOCAL_RANK  : ', LOCAL_RANK)
print ('WORLD_SIZE  : ', WORLD_SIZE)
print ('WORLD_RANK  : ', WORLD_RANK)
print ('cuda device : ', torch.cuda.device_count())
print ('pytorch version : ', torch.__version__)
print ('nccl version : ', print(torch.cuda.nccl.version()))
print ('----------------------')    


def run(backend):
    tensor = torch.zeros(1000000)
    
    # Need to put tensor on a GPU device for nccl backend
    if backend == 'nccl':
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        tensor = tensor.to(device)

    torch.cuda.synchronize()
    start_time = time.time()

    if WORLD_RANK == 0:
        for rank_recv in range(1, WORLD_SIZE):
            dist.send(tensor=tensor, dst=rank_recv)
            print('worker_{} sent data to Rank {}\n'.format(0, rank_recv))
    else:
        dist.recv(tensor=tensor, src=0)
        print('worker_{} has received data from rank {}\n'.format(WORLD_RANK, 0))

    torch.cuda.synchronize()  # Ensure all operations completed
    end_time = time.time()

    # Calculating the duration of the send/receive operation
    print(f'Rank {WORLD_RANK}: Operation took {end_time - start_time:.6f} seconds')


def init_processes(backend):
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    run(backend)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--backend", type=str, default="nccl", choices=['nccl', 'gloo'])
    args = parser.parse_args()

    init_processes(backend=args.backend)
