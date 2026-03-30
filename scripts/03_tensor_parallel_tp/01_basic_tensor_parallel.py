import os
import sys

# Add repo root to path so `from utils...` works
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.distributed as dist

from utils.distributed import init_distributed, cleanup_distributed

rank, world_size, local_rank = init_distributed()
print(f"Running example on rank={rank} in a world of size={world_size}")

# --- Define shard and replicate groups ---
num_node_devices = torch.cuda.device_count()
if num_node_devices < 2:
    raise RuntimeError("Need at least 2 GPUs per node for this 2D parallel example.")

# Example: split local GPUs into two shard groups
half = num_node_devices // 2
shard_rank_lists = [list(range(0, half)), list(range(half, num_node_devices))]
shard_groups = [dist.new_group(ranks) for ranks in shard_rank_lists]

current_shard_group = None
for ranks, group in zip(shard_rank_lists, shard_groups):
    if (rank % num_node_devices) in ranks:
        current_shard_group = group
        break

# Example: replicate groups (e.g. (0,4), (1,5), ...)
shard_factor = half
current_replicate_group = None
for i in range(half):
    replicate_ranks = list(range(i, world_size, shard_factor))
    group = dist.new_group(replicate_ranks)
    if rank in replicate_ranks:
        current_replicate_group = group

print(f"Rank {rank}: shard_group={shard_rank_lists}, replicate_group created.")

# --- Verify communication works ---
tensor = torch.tensor([rank], dtype=torch.float32, device="cuda")
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
print(f"Rank {rank} sees all_reduce result: {tensor.item()}")

dist.barrier()
cleanup_distributed()

