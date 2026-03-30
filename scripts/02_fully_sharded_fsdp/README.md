# PyTorch Fully Sharded Data Parallel (FSDP)

PyTorch Fully Sharded Data Parallel (FSDP) speeds up model training by
parallelizing training data **and** sharding model parameters, optimizer states,
and gradients across multiple GPUs.

If your model does not fit on a single GPU, you can use FSDP and request more
GPUs to reduce the memory footprint on each GPU. The model parameters are split
between the GPUs and each training process receives a different subset of
training data. Model updates from each device are broadcast across devices,
resulting in the same model on all devices.

For a complete overview with examples, see the
[PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html).

## How FSDP Works

FSDP keeps parameters sharded at rest. When a layer needs to execute,
FSDP all-gathers the full parameters, runs the operation, then frees them.
During the backward pass, it all-gathers params again, computes local
gradients, reduce-scatters gradients across GPUs, frees full params, and
each GPU updates only its shard of the parameters.

## Environment Setup

For running FSDP on Derecho, activate the shared conda environment or your own
environment with PyTorch and torchvision installed:

```bash
module load nvhpc cuda cray-mpich conda
conda activate pytorch-derecho
```

Or use the shared PBS setup that handles this automatically (see
[PBS Job Script](#pbs-job-script) below).

## How FSDP Differs from Single-GPU Training

There are **6 key differences** between FSDP and single-GPU training:

### 1. FSDP Setup — Initialize a Process Group

FSDP creates a process group and sets the local device. This is called at the
start of `main()`. Our shared utility handles launcher detection automatically
(torchrun, OpenMPI, Cray MPICH):

```python
from utils.distributed import init_distributed, cleanup_distributed

world_rank, world_size, local_rank = init_distributed()
```

### 2. Wrap the Model in FSDP

Instead of using the model directly, wrap it with `FullyShardedDataParallel`.
An auto-wrap policy controls which submodules get their own FSDP unit:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy, min_num_params=1e5
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=mixed_precision,   # optional, see below
    device_id=local_rank,
)
```

### 3. Use DistributedSampler to Load Data

Each GPU must receive a different subset of training data. The
`DistributedSampler` partitions the dataset across ranks:

```python
from torch.utils.data.distributed import DistributedSampler

train_sampler = DistributedSampler(
    train_dataset, num_replicas=world_size, rank=rank
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,    # replaces shuffle=True
    num_workers=2,
    pin_memory=True,
)
```

### 4. Set the Epoch on the Sampler Each Epoch

This ensures each epoch uses a different data shuffle. Without this, every
epoch would see the same data order:

```python
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    # ... training loop ...
```

### 5. Destroy the Process Group After Training

Clean up distributed resources when training is complete:

```python
cleanup_distributed()
```

### 6. Track Global vs Local Rank Separately

`local_rank` identifies the GPU on the current node (0-3 on a 4-GPU node).
`world_rank` identifies the process across all nodes. Use `world_rank == 0`
for logging and checkpointing:

```python
world_rank, world_size, local_rank = init_distributed()

device = torch.device(f"cuda:{local_rank}")

if world_rank == 0:
    print("Training started")   # only print once across all GPUs
```

## Sharding Strategies

| Strategy | What's Sharded | Memory Savings | Communication |
|----------|---------------|----------------|---------------|
| `FULL_SHARD` | Params + grads + optimizer | Best | Most |
| `SHARD_GRAD_OP` | Grads + optimizer only | Good | Less |
| `NO_SHARD` | Nothing (same as DDP) | None | Least |
| `HYBRID_SHARD` | Full shard within node, replicate across | Good | Balanced |

## Mixed Precision

BFloat16 mixed precision reduces memory further and speeds up training on
Derecho's A100 GPUs:

```python
from torch.distributed.fsdp import MixedPrecision

mp = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)
model = FSDP(model, mixed_precision=mp)
```

Enable it with `--use-amp` when running the example script.

## Running the Example

### Files

| Script | Description |
|--------|-------------|
| `resnet_fsdp_training.py` | FSDP training with ResNet-18 on CIFAR-10 |
| `run_fsdp.sh` | PBS job script for Derecho |

### Single Node (4 GPUs) — torchrun

```bash
torchrun --standalone --nproc_per_node=4 resnet_fsdp_training.py
torchrun --standalone --nproc_per_node=4 resnet_fsdp_training.py --use-amp
```

### Multi-Node — mpiexec

```bash
mpiexec -n 8 --ppn 4 --cpu-bind none python resnet_fsdp_training.py
mpiexec -n 8 --ppn 4 --cpu-bind none python resnet_fsdp_training.py --use-amp
```

### PBS Job Script

Submit to the Derecho job scheduler:

```bash
qsub run_fsdp.sh
```

The job script `run_fsdp.sh` is a self-contained template with module
loading, conda activation, NCCL configuration for Slingshot, and node
discovery all inlined — copy it and adjust for your own job. For
multi-node runs, change the `select` line:

```bash
#PBS -l select=2:ncpus=64:ngpus=4:mem=480GB
```

## When to Use FSDP vs DDP

**Use FSDP when:**
- Model parameters exceed single-GPU memory
- Training large language models or vision transformers
- You need memory-efficient multi-GPU training

**Use DDP instead when:**
- Model fits comfortably on one GPU (DDP has less communication overhead)
- Communication bandwidth is limited (FSDP requires more all-gather traffic)

## References

- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [FSDP Paper (Zhao et al. 2023)](https://arxiv.org/abs/2304.11277)
