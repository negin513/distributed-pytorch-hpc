# Chapter 5: Fully Sharded Data Parallel (FSDP)

DDP keeps a full copy of the model on every GPU. That works until your
model is too large for one GPU's memory. FSDP solves this by **sharding**
parameters, gradients, and optimizer state across GPUs — each GPU only
stores 1/N of everything.

## The Problem with DDP

With DDP and 4 GPUs, you have 4 complete copies of the model:

```
DDP with 4 GPUs (7B model, FP32 + Adam):

GPU 0: [full model 112 GB]  ← doesn't fit on 40 GB A100
GPU 1: [full model 112 GB]
GPU 2: [full model 112 GB]
GPU 3: [full model 112 GB]

Total memory used: 4 × 112 GB = 448 GB
Unique data: 112 GB
Redundancy: 4×
```

FSDP eliminates this redundancy.

## How FSDP Works

FSDP shards parameters across GPUs. Each GPU only stores its 1/N shard.
When a layer needs the full parameters (for forward or backward), FSDP
temporarily gathers them, uses them, and discards the non-local parts.

```
FSDP Lifecycle (4 GPUs):

1. At rest — each GPU holds 1/4 of params:
   GPU 0: [shard 0]
   GPU 1: [shard 1]
   GPU 2: [shard 2]
   GPU 3: [shard 3]

2. Before forward — all-gather to reconstruct full params:
   GPU 0: [shard 0 | shard 1 | shard 2 | shard 3]  (temporary)
   GPU 1: [shard 0 | shard 1 | shard 2 | shard 3]  (temporary)
   ...

3. Compute forward pass (using full params)

4. After forward — discard non-local shards:
   GPU 0: [shard 0]  (back to 1/4)

5. Before backward — all-gather again

6. After backward — reduce-scatter gradients:
   GPU 0: [grad shard 0]  (already reduced + sharded)

7. Optimizer step — each GPU updates only its shard
```

### Memory comparison

```
FSDP with 4 GPUs (7B model, FP32 + Adam):

GPU 0: [1/4 model ≈ 28 GB]  ← fits on 40 GB A100!
GPU 1: [1/4 model ≈ 28 GB]
GPU 2: [1/4 model ≈ 28 GB]
GPU 3: [1/4 model ≈ 28 GB]

Total memory used: 4 × 28 GB = 112 GB
Unique data: 112 GB
Redundancy: 1× (none!)
```

## From DDP to FSDP

The code change is small. Replace `DDP(model)` with `FSDP(model)` and
add a wrapping policy:

```python
# DDP version:
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[local_rank])

# FSDP version:
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy, min_num_params=100_000
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    device_id=local_rank,
)
```

The training loop stays the same — `zero_grad → forward → backward → step`.

## Sharding Strategies

FSDP offers three strategies with different memory/speed tradeoffs:

| Strategy | What's Sharded | Memory | Communication | When to Use |
|----------|---------------|--------|---------------|-------------|
| `FULL_SHARD` | Params + gradients + optimizer | Lowest | Highest (all-gather + reduce-scatter) | Model barely fits across all GPUs |
| `SHARD_GRAD_OP` | Gradients + optimizer | Medium | Medium (reduce-scatter only) | Model fits but optimizer doesn't |
| `NO_SHARD` | Nothing (like DDP) | Highest | Lowest (all-reduce only) | Debugging / comparison |

```python
from torch.distributed.fsdp import ShardingStrategy

# Maximum memory savings
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, ...)

# Less communication, more memory
model = FSDP(model, sharding_strategy=ShardingStrategy.SHARD_GRAD_OP, ...)
```

## Mixed Precision

FSDP works with mixed precision to further reduce memory. On A100 GPUs,
BFloat16 is the preferred format:

```python
from torch.distributed.fsdp import MixedPrecision

mixed_precision = MixedPrecision(
    param_dtype=torch.bfloat16,     # compute in BF16
    reduce_dtype=torch.bfloat16,    # communicate in BF16
    buffer_dtype=torch.bfloat16,    # buffers in BF16
)

model = FSDP(
    model,
    mixed_precision=mixed_precision,
    ...
)
```

This halves memory for parameters during computation and halves the data
sent during all-gather and reduce-scatter.

## Wrapping Policies

FSDP doesn't shard the model as a single unit — it wraps **sub-modules**
individually. The wrapping policy determines which modules get their own
FSDP wrapper:

### Size-based (simple)

Wrap any module with more than N parameters:

```python
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy, min_num_params=100_000
)
```

### Module-type based (precise)

Wrap specific module types (common for transformers):

```python
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

# Wrap each transformer block individually
auto_wrap_policy = ModuleWrapPolicy({TransformerBlock})
```

Wrapping at the right granularity matters: too coarse and you lose
sharding benefit; too fine and communication overhead dominates.

## FSDP vs DDP: When to Switch

| Situation | Recommendation |
|-----------|---------------|
| Model fits on 1 GPU | Use DDP (simpler, faster) |
| Model fits but training is slow | Use DDP with more GPUs |
| Optimizer state doesn't fit | Try `SHARD_GRAD_OP` |
| Model doesn't fit on 1 GPU | Use `FULL_SHARD` |
| Model doesn't fit across all GPUs | Add TP (Chapter 6) or PP (Chapter 7) |

## Running the Example

```bash
# Single node, 4 GPUs
torchrun --standalone --nproc_per_node=4 \
    scripts/02_fully_sharded_fsdp/resnet_fsdp_training.py

# With mixed precision
torchrun --standalone --nproc_per_node=4 \
    scripts/02_fully_sharded_fsdp/resnet_fsdp_training.py --use-amp

# Via PBS job script
qsub scripts/02_fully_sharded_fsdp/run_fsdp.sh
```

**See also:**
- [`scripts/02_fully_sharded_fsdp/resnet_fsdp_training.py`](../../scripts/02_fully_sharded_fsdp/resnet_fsdp_training.py) — FSDP training with ResNet-18 on CIFAR-10
- [`scripts/02_fully_sharded_fsdp/README.md`](../../scripts/02_fully_sharded_fsdp/README.md) — deep dive on FSDP

## What's Next?

FSDP shards entire parameters across all GPUs. But what if a single
layer's weight matrix is too large? Tensor Parallelism splits individual
layers across GPUs.

**Next:** [Chapter 6 — Tensor Parallel](06_tensor_parallel.md)
