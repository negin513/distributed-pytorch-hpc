# Chapter 4: Data Parallel (DDP)

Data Parallel is the most common distributed training strategy and usually
the first one you should try. The idea is simple: every GPU has a complete
copy of the model, but each processes a different slice of the data.

## From Single-GPU to DDP in 5 Changes

Starting from the single-GPU script in Chapter 1, here are the only
changes needed:

```python
# ── Change 1: Initialize the process group ──
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# ── Change 2: Use local_rank for device ──
device = torch.device(f"cuda:{local_rank}")
model = model.to(device)

# ── Change 3: Wrap model with DDP ──
model = DDP(model, device_ids=[local_rank])

# ── Change 4: Use DistributedSampler instead of shuffle=True ──
train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=64,
                          sampler=train_sampler, pin_memory=True)

# ── Change 5: Set epoch on the sampler ──
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)    # ensures different shuffling each epoch
    for data, target in train_loader:
        # ... same training loop as before
```

That's it. Your training loop (`zero_grad → forward → backward → step`)
stays exactly the same.

## How DDP Works

```
Step 1: Each GPU has a full model copy
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Model    │  │ Model    │  │ Model    │  │ Model    │
│ (copy)   │  │ (copy)   │  │ (copy)   │  │ (copy)   │
│          │  │          │  │          │  │          │
│ Batch 0  │  │ Batch 1  │  │ Batch 2  │  │ Batch 3  │
│ GPU 0    │  │ GPU 1    │  │ GPU 2    │  │ GPU 3    │
└──────────┘  └──────────┘  └──────────┘  └──────────┘

Step 2: Each GPU computes forward + backward on its own batch
         (produces local gradients)

Step 3: All-reduce averages gradients across all GPUs
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ ∇avg     │  │ ∇avg     │  │ ∇avg     │  │ ∇avg     │
│ = (∇0 +  │  │ = (∇0 +  │  │ = (∇0 +  │  │ = (∇0 +  │
│  ∇1+∇2+  │  │  ∇1+∇2+  │  │  ∇1+∇2+  │  │  ∇1+∇2+  │
│  ∇3) / 4 │  │  ∇3) / 4 │  │  ∇3) / 4 │  │  ∇3) / 4 │
└──────────┘  └──────────┘  └──────────┘  └──────────┘

Step 4: Each GPU does optimizer.step() with identical gradients
         → models stay in sync
```

The all-reduce in Step 3 is the only communication. DDP overlaps it with
the backward pass — as soon as a gradient bucket is ready, the all-reduce
starts, even while later layers are still computing.

## Effective Batch Size

With DDP, each GPU processes `batch_size` samples per step. The effective
batch size is:

```
effective_batch_size = per_gpu_batch_size × world_size
```

If you use `batch_size=64` on 4 GPUs, your effective batch size is 256.
This matters for learning rate scheduling — you may need to scale the
learning rate accordingly (linear scaling rule: `lr × world_size`).

## DistributedSampler

The `DistributedSampler` ensures each GPU sees a **non-overlapping**
subset of the data:

```
Dataset: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

GPU 0 (rank 0): [0, 4,  8]    ← every 4th sample starting at 0
GPU 1 (rank 1): [1, 5,  9]
GPU 2 (rank 2): [2, 6, 10]
GPU 3 (rank 3): [3, 7, 11]
```

Two important details:

1. **Set `shuffle=False` in DataLoader** — the sampler handles shuffling.
2. **Call `sampler.set_epoch(epoch)`** at the start of each epoch.
   Without this, every epoch uses the same shuffle order, which hurts
   convergence.

## Common Pitfalls

### Printing on all ranks

Without guarding print statements, you get 4 copies of every log line:

```python
# Bad: prints on all GPUs
print(f"Loss: {loss.item()}")

# Good: only rank 0 prints
if dist.get_rank() == 0:
    print(f"Loss: {loss.item()}")
```

### Forgetting set_epoch()

```python
# Bad: same data order every epoch
for epoch in range(num_epochs):
    for data, target in train_loader:
        ...

# Good: different shuffle each epoch
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    for data, target in train_loader:
        ...
```

### Saving checkpoints on all ranks

Only save on rank 0. All ranks have identical weights, so saving on every
GPU wastes storage and can cause file corruption:

```python
if dist.get_rank() == 0:
    torch.save(model.module.state_dict(), "checkpoint.pt")
```

Note: access `model.module` to get the unwrapped model state dict.

### Not calling cleanup

Always destroy the process group when training ends:

```python
dist.destroy_process_group()
```

## Performance: Gradient Bucketing

DDP doesn't all-reduce each parameter individually. It groups parameters
into **buckets** (default 25 MB) and all-reduces entire buckets at once.
This reduces the number of communication calls and allows overlap with
computation:

```
Backward pass timeline:

Layer 10:  [compute grad]
Layer 9:   [compute grad]──┐
Layer 8:   [compute grad]  │ Bucket 1 ready
Layer 7:   [compute grad]──┤──► [all-reduce bucket 1]
Layer 6:   [compute grad]  │
Layer 5:   [compute grad]──┤ Bucket 2 ready
Layer 4:   [compute grad]  ├──► [all-reduce bucket 2]
Layer 3:   [compute grad]  │
Layer 2:   [compute grad]──┤ Bucket 3 ready
Layer 1:   [compute grad]  └──► [all-reduce bucket 3]
```

The overlap means communication is partially hidden behind computation.

## When to Use DDP

**Use DDP when:**
- Your model fits on a single GPU
- You want to train faster by using more GPUs
- You want near-linear scaling with GPU count

**Move beyond DDP when:**
- Your model doesn't fit on a single GPU → Chapter 5 (FSDP)
- Individual layers are too large → Chapter 6 (TP)
- You have hundreds of layers → Chapter 7 (PP)

## Running the Examples

```bash
# Single node, 4 GPUs
torchrun --standalone --nproc_per_node=4 \
    scripts/01_data_parallel_ddp/multinode_ddp_basic.py

# Multi-node (via PBS job script)
qsub scripts/01_data_parallel_ddp/torchrun_multigpu_ddp.sh
```

**See also:**
- [`scripts/01_data_parallel_ddp/multinode_ddp_basic.py`](../../scripts/01_data_parallel_ddp/multinode_ddp_basic.py) — minimal DDP example with synthetic data
- [`scripts/01_data_parallel_ddp/distributed_dataloader.py`](../../scripts/01_data_parallel_ddp/distributed_dataloader.py) — DistributedSampler patterns
- [`scripts/01_data_parallel_ddp/README.md`](../../scripts/01_data_parallel_ddp/README.md) — deep dive on DDP

## What's Next?

DDP requires every GPU to hold the full model. When that's no longer
possible, FSDP shards the model itself across GPUs.

**Next:** [Chapter 5 — Fully Sharded Data Parallel (FSDP)](05_fully_sharded_fsdp.md)
