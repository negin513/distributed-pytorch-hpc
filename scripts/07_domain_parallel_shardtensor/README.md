# 07 - Domain Parallelism with Halo Exchange

Domain parallelism splits **spatial data** (not model weights) across GPUs.
This is critical for scientific AI where high-resolution inputs (weather grids,
medical images, CFD meshes) don't fit in a single GPU's memory even at batch size 1.

These examples use **pure PyTorch** -- no external dependencies like PhysicsNeMo.
The same concepts apply to NVIDIA's ShardTensor framework, which automates
what we build explicitly here.

## Prerequisites

Only PyTorch with NCCL backend (no extra packages needed).

```bash
conda activate /glade/work/negins/conda-envs/torch28-nccl221
```

## Examples (progressive complexity)

| Script | GPUs | What it demonstrates |
|--------|------|---------------------|
| `01_why_splitting_fails.py` | 1 | Why naive spatial splitting gives wrong convolution results |
| `04_domain_parallel_with_fsdp.py` | 2+ | Full training with manual halo exchange, optionally combined with FSDP |

## Quick start

```bash
cd /glade/work/negins/distributed-pytorch-hpc/scripts/07_domain_parallel_shardtensor

# Single GPU demo -- no distributed setup needed
python 01_why_splitting_fails.py
```

### Single node (4 GPUs) with mpiexec

```bash
# Pure domain parallelism (4-way spatial split)
mpiexec -n 4 --ppn 4 --cpu-bind none python 04_domain_parallel_with_fsdp.py --domain-size 4

# Domain + FSDP hybrid (2x2 mesh)
mpiexec -n 4 --ppn 4 --cpu-bind none python 04_domain_parallel_with_fsdp.py --domain-size 2 --fsdp-size 2

# Larger image
mpiexec -n 4 --ppn 4 --cpu-bind none python 04_domain_parallel_with_fsdp.py --domain-size 4 --image-size 2048
```

### Single node (4 GPUs) with torchrun

```bash
torchrun --standalone --nproc_per_node=4 04_domain_parallel_with_fsdp.py --domain-size 4
torchrun --standalone --nproc_per_node=4 04_domain_parallel_with_fsdp.py --domain-size 2 --fsdp-size 2
```

### Multi-node (8 GPUs, 2 nodes) with mpiexec

```bash
export NCCL_SOCKET_IFNAME=hsn
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=VERSION

# 4-way domain x 2-way FSDP
mpiexec -n 8 --ppn 4 --cpu-bind none python 04_domain_parallel_with_fsdp.py \
    --domain-size 4 --fsdp-size 2 --image-size 1024 --batch-size 8 --num-steps 10
```

### PBS on Derecho

```bash
qsub run_domain_parallel.sh
```

### Constraint

`domain-size * fsdp-size` must equal the total number of GPUs, and `image-size`
must be divisible by `domain-size`.

## When to use domain parallelism

**Use it when:**
- Input data at batch_size=1 **exceeds GPU memory** (activations dominate)
- Model uses spatial ops: convolutions, norms, pooling, attention
- Fast GPU interconnect available (NVLink, Slingshot)

**Don't use it when:**
- Data fits on one GPU -- use DDP instead (less overhead)
- Model is dominated by tiny kernels -- halo exchange overhead hurts

## The Problem: Why Naive Splitting Fails

A 3x3 convolution needs to see 1 neighboring pixel on each side.
When you split a 1024x1024 image in half along the height dimension and
apply the convolution independently, the output is **wrong at rows 511-512**
(the split boundary). The kernel can't see across the split.

```
Full image:   [...row 510, row 511 | row 512, row 513...]
                                   ^
                              split point

GPU 0 has:    [...row 510, row 511]     <-- missing row 512 for the kernel
GPU 1 has:    [row 512, row 513...]     <-- missing row 511 for the kernel
```

`01_why_splitting_fails.py` demonstrates this on a single GPU.

## The Solution: Halo Exchange

Before each convolution, neighboring GPUs exchange **halo rows** (also called
ghost cells). For a 3x3 kernel, each GPU needs 1 extra row from each neighbor:

```
Before halo exchange:
  GPU 0: [rows 0-511]              GPU 1: [rows 512-1023]

After halo exchange:
  GPU 0: [rows 0-511, row 512]     GPU 1: [row 511, rows 512-1023]
                      ^^^^^^^^             ^^^^^^^^
                      received             received
                      from GPU 1           from GPU 0
```

After the convolution, the extra output rows from the halo padding are
trimmed so the output has the same local spatial size as the input.

## How `04_domain_parallel_with_fsdp.py` Works

The script implements three key components:

### 1. HaloExchange (custom autograd.Function)

A differentiable operation that uses NCCL point-to-point (`P2POp`) sends
and receives to exchange boundary rows between neighboring GPUs.

**Forward pass:** Each GPU sends its boundary rows to its neighbors and
receives theirs, then concatenates them as padding:

```
Input:  [B, C, H_local, W]
Output: [B, C, H_local + halo_top + halo_bot, W]
```

**Backward pass:** Gradient from halo regions is sent back to the GPU
that owns those rows and added to its local gradient. This ensures
gradients are correct across the domain boundary.

### 2. DomainParallelConv2d

Wraps a standard `nn.Conv2d` with halo exchange:

```
forward(x):
    1. halo_exchange(x)           # pad height with neighbor rows via NCCL
    2. F.pad(x, width_padding)    # pad width normally (not sharded)
    3. conv(x)                    # standard conv, no built-in padding
```

The conv is created with `padding=0` because all padding is handled
explicitly -- halo exchange for the sharded dimension (height),
`F.pad` for the non-sharded dimension (width).

### 3. GroupNorm instead of BatchNorm

BatchNorm computes mean/variance over `(B, H, W)`. With sharded data,
each GPU only sees part of H, so a correct BatchNorm would require an
**allreduce** across all domain GPUs at every layer (expensive).

GroupNorm computes statistics over groups of channels per-sample, which
works correctly on each GPU's local spatial slice without communication.

### 4. 2D Process Group Mesh

GPUs are arranged in a 2D grid: `[fsdp_size x domain_size]`

```
Example: 4 GPUs, domain=2, fsdp=2

                    domain_dim -->
                 shard 0    shard 1
  fsdp_dim   +-----------+-----------+
  replica 0  |   GPU 0   |   GPU 1   |   <- domain group {0,1}: split image spatially
             +-----------+-----------+
  replica 1  |   GPU 2   |   GPU 3   |   <- domain group {2,3}: split image spatially
             +-----------+-----------+
                  |             |
             fsdp group    fsdp group
              {0,2}          {1,3}
              share          share
              weights        weights
```

- **Domain groups (rows):** GPUs that split the image spatially and exchange halos
- **FSDP groups (columns):** GPUs that shard model weights for data parallelism

Each GPU holds:
- A **spatial slice** of the image (1/domain_size of the height)
- A **shard of the model weights** (via FSDP, 1/fsdp_size of the parameters)
- A **fraction of the batch** (1/fsdp_size of the global batch)

### Memory reduction

With `domain=2, fsdp=2` on a 1024x1024 image with batch_size=4:
- Without domain parallelism: each GPU holds `[4, 3, 1024, 1024]` = 50.3 MB input
- With domain+FSDP: each GPU holds `[2, 3, 512, 1024]` = 12.6 MB input (**4x reduction**)
- Activation memory savings are even larger (proportional to depth x spatial size)

## Comparison with PhysicsNeMo ShardTensor

| | Manual halo exchange (this example) | PhysicsNeMo ShardTensor |
|---|---|---|
| Dependencies | Pure PyTorch | Requires `physicsnemo` |
| Conv2d halo exchange | Explicit `P2POp` | Automatic (registered dispatch) |
| BatchNorm | Must use GroupNorm or manual allreduce | Has sharded BatchNorm implementation |
| Backward pass | Custom `autograd.Function` | Automatic via DTensor dispatch |
| Uneven data (meshes, point clouds) | Not supported | Supported (tracks per-rank shapes) |
| Effort to add new layers | Write halo exchange per op | Register sharding strategy |

For production use with many layer types, ShardTensor is more convenient.
For learning the concepts or using only convolutions, manual halo exchange
works with zero dependencies.
