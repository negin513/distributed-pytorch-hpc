# Chapter 10: Domain Parallel

Domain Parallelism is fundamentally different from every other strategy
in this guide. Instead of splitting the **model**, it splits the **input data
spatially** — each GPU processes a tile of a large grid, image, or mesh.

This is critical for scientific computing and SciML: climate models, weather
prediction, computational fluid dynamics, and medical imaging — where the
input is a high-resolution spatial field that exceeds a single GPU's memory.

## Why Scientific AI Memory Is Different

In LLMs, GPU memory is dominated by model parameters (billions of weights).
In scientific AI, the situation is reversed —> models are often small (a few million parameters), but **activations dominate memory** because the input
data is spatially massive.

In scientific AI, the bottleneck is often GPU memory capacity, not compute. You can have a model that runs at 100 TFLOPS but still OOMs because the input data is too large. Domain parallelism addresses this by never coalescing the full input on a single GPU.

!!! note "What goes on to the GPU memory?"
    During training, GPU memory is consumed by four things:
    1. **Model parameters** — small for most scientific models
    2. **Active data** (inputs/outputs) — large at high resolution
    3. **Optimizer states** (gradients, moments) — proportional to parameters
    4. **Intermediate activations** — saved for the backward pass, proportional to *both* model depth and input resolution

    As layers stack up, activation memory grows with depth *and* resolution.
    A U-Net on a 1024x1024 grid can easily require 10-100x more activation
    memory than parameter memory. This is why DDP and FSDP alone aren't
    enough — they shard parameters and gradients, not activations. Domain
    parallelism is the only strategy that addresses activation memory.

## Why Domain Parallel?

Consider a global weather model at 0.25-degree resolution:

```
Grid: 1440 × 720 pixels × 100+ vertical levels × multiple variables

A single forward pass through a U-Net on this grid might need 100+ GB
of activation memory, which is far more than a single GPU.
```

Domain parallelism splits the grid:

```
Full grid (1440 × 720):

┌───────────┬───────────┐
│           │           │
│  GPU 0    │  GPU 1    │
│  (720×360)│  (720×360)│
│           │           │
├───────────┼───────────┤
│           │           │
│  GPU 2    │  GPU 3    │
│  (720×360)│  (720×360)│
│           │           │
└───────────┴───────────┘

Each GPU processes 1/4 of the spatial domain.
```

### Why Naive Splitting Fails

But, you can't just slice the grid and run convolutions independently. At tile
boundaries, convolutions produce incorrect results because they don't have
access to neighboring pixels:

```
Naive split (3×3 convolution):

GPU 0 tile              GPU 1 tile
┌─────────┐            ┌─────────┐
│ · · · · │ ← border → │ · · · · │
│ · · · · │            │ · · · · │
│ · · · x │            │ x · · · │
└─────────┘            └─────────┘
        ▲                ▲
        │                │
    This pixel           This pixel
    needs data           needs data
    from GPU 1           from GPU 0

Without the neighbor's data, border pixels get wrong values.
```

You can demonstrate this on a single device in PyTorch:

```python
import torch

full_image = torch.randn(1, 8, 1024, 1024)
left_image  = full_image[:, :, :512, :]
right_image = full_image[:, :, 512:, :]

conv = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)

full_output  = conv(full_image)
left_output  = conv(left_image)
right_output = conv(right_image)
recombined   = torch.cat([left_output, right_output], dim=2)

torch.allclose(full_output, recombined)  # False!
```

Inspecting where the outputs disagree reveals the problem is exactly at
pixels 511 and 512 along the height dimension — right where the data was
split. The convolution can't see across the border.

The fix is to exchange the missing border row before convolving:

```python
# Pad each half with 1 row from the neighbor (this is a halo exchange)
padded_left  = torch.cat([left_image, right_image[:, :, 0:1, :]], dim=2)
padded_right = torch.cat([left_image[:, :, -1:, :], right_image], dim=2)

# Conv on padded data, then trim the extra output pixels
left_output  = conv(padded_left)[:, :, :-1, :]
right_output = conv(padded_right)[:, :, 1:, :]
recombined   = torch.cat([left_output, right_output], dim=2)

torch.allclose(full_output, recombined)  # True!
```

This manual padding is exactly what **halo exchange** automates across GPUs.

### Halo Exchange

Before each convolution, GPUs exchange a thin border of pixels with their
neighbors:

```
Step 1: Each GPU adds a "halo" region around its tile

              halo
          ┌─── ▼ ───┐
GPU 0:    │ h │ data │ h │     h = halo (received from neighbor)
          └─────────────┘

Step 2: Exchange halos with neighbors via point-to-point communication

GPU 0          GPU 1          GPU 2          GPU 3
[data|h] ──► [h|data|h] ──► [h|data|h] ──► [h|data]
       ◄──          ◄──           ◄──

Step 3: Run convolution on padded tile (including halos)

Step 4: Discard halos, keep only the interior result
```

The halo width equals the convolution's padding (or receptive field for
multi-layer blocks). A 3×3 conv needs 1 pixel of halo; a 5×5 conv needs
2.

```
2D halo exchange (4 GPUs in a 2×2 grid):

         ┌──────────────┬──────────────┐
         │    GPU 0     │    GPU 1     │
         │              │              │
         │         ◄───►│              │  ← horizontal halo
         │    ▲         │    ▲         │
         │    │         │    │         │
         ├────┼─────────┼────┼─────────┤
         │    ▼         │    ▼         │  ← vertical halo
         │              │              │
         │         ◄───►│              │
         │    GPU 2     │    GPU 3     │
         └──────────────┴──────────────┘

Each GPU exchanges halos with up to 4 neighbors (N, S, E, W).
Corner halos can be handled with diagonal exchanges or two rounds.
```

Gradients also require communication across the split boundary during
`backward()`. This adds overhead, but the memory savings from never
coalescing the full tensor on a single GPU make it worthwhile.

## GroupNorm Instead of BatchNorm

BatchNorm computes statistics across the batch dimension, which requires
communication across all GPUs. For domain parallelism, use **GroupNorm**
instead — it computes statistics within each sample independently:

```
BatchNorm: mean/var across batch → needs all-reduce across GPUs
GroupNorm: mean/var within groups of channels → fully local
```

This avoids a costly all-reduce at every normalization layer.

## Domain Parallel vs. Other Approaches

Domain parallelism isn't the only way to handle large inputs. Here's how
the alternatives compare:

| Approach | What it does | Strengths | Limitations |
|----------|-------------|-----------|-------------|
| **Domain Parallel** | Splits spatial input across GPUs | Scales both memory and compute; works with complex architectures (U-Net) | Halo communication overhead per layer |
| **Pipeline Parallel** | Splits model layers across GPUs | Low communication (only between stages) | Pipeline bubble; GPUs idle while waiting; hard for skip connections (U-Net) |
| **Activation Checkpointing** | Offloads activations to CPU, recomputes during backward | No code changes to model; works with DDP | Limited by CPU-GPU bandwidth; recompute cost |
| **DDP** | Replicates model, splits batch | Simple; high throughput | Doesn't help if `batch_size=1` already OOMs |

**Pipeline parallelism** divides the model by layers — GPU 0 runs layers
0-9, GPU 1 runs layers 10-19, etc. It scales GPU memory, but not compute:
while GPU 0 is active, other GPUs wait. It also doesn't work well for
architectures with skip connections (like U-Net) where concatenation spans
the down/up sampling paths.

**Activation checkpointing** moves intermediate activations from GPU to CPU
during the forward pass and restores them during backward. It can be limited
by CPU-GPU transfer speeds and doesn't scale compute across GPUs.

Domain parallelism is the best fit when your input is so large that even
`batch_size=1` doesn't fit, and your model uses spatial operations
(convolutions, normalizations, attention, pooling).

> If your model comfortably fits `batch_size=1` training, DDP will be
> more efficient. Domain parallelism shines when the data is the bottleneck,
> not the model.

## Combining Domain Parallel with FSDP

For large models on large grids, you can combine domain parallelism with
FSDP using a 2D mesh:

```
2D mesh: [FSDP × Domain]

4 nodes × 4 GPUs = 16 GPUs

FSDP groups (model sharding):
  [0, 4, 8, 12]  [1, 5, 9, 13]  [2, 6, 10, 14]  [3, 7, 11, 15]

Domain groups (spatial splitting):
  [0, 1, 2, 3]  [4, 5, 6, 7]  [8, 9, 10, 11]  [12, 13, 14, 15]
```

Each domain group handles a different spatial tile, while FSDP shards
the model parameters across domain groups.

## ShardTensor: Production Domain Parallelism

For production workloads, NVIDIA's
[ShardTensor](https://docs.nvidia.com/physicsnemo/user-guide/latest/physicsnemo-distributed/domain-parallelism/shard-tensor.html)
(built on PyTorch's DTensor) automates what we did manually above.
Instead of hand-coding halo exchanges and gradient communication:

- **Automatic halo exchange** — operations are intercepted at the functional
  level; communication happens transparently without manual padding or send/recv
- **Correct gradients** — `mean().backward()` on a ShardTensor automatically
  distributes gradients to their proper sharding
- **Irregular data support** — unlike DTensor's uniform `torch.chunk`,
  ShardTensor handles meshes, point clouds, and unevenly-distributed domains

Under the hood, ShardTensor extends PyTorch's DTensor with:

- A specification that tracks the shape of each local tensor along sharding
  axes (critical for non-uniform data like point clouds)
- A dispatcher that intercepts operations at the functional level (higher
  than DTensor's dispatch level), falling back to DTensor when no custom
  implementation exists
- Dedicated `sum` and `mean` reductions that correctly intercept and
  distribute gradients

Domain parallelism with ShardTensor performs best when:

- GPU kernels are **large** (big input data) — the communication-to-compute ratio stays small
- GPU kernels are **non-blocking** — the slightly higher overhead of domain parallelism still fills the GPU queue efficiently

For small kernels or latency-bound models, pipeline parallelism or
activation checkpointing may be more efficient.

**Further reading:**

- [Domain Parallelism and ShardTensor](https://docs.nvidia.com/physicsnemo/user-guide/latest/physicsnemo-distributed/domain-parallelism/shard-tensor.html)
- [Implementing New Layers for ShardTensor](https://docs.nvidia.com/physicsnemo/user-guide/latest/physicsnemo-distributed/domain-parallelism/implementing-new-layers.html)
- [Domain Decomposition, ShardTensor, and FSDP Tutorial](https://docs.nvidia.com/physicsnemo/user-guide/latest/physicsnemo-distributed/domain-parallelism/fsdp-and-shard-tensor.html)

## Running the Examples

Start with the "why splitting fails" demo to build intuition:

```bash
# 1. See why naive splitting produces boundary artifacts
torchrun --standalone --nproc_per_node=4 \
    scripts/07_domain_parallel_shardtensor/01_why_splitting_fails.py

# 2. Domain-parallel convolution with halo exchange
torchrun --standalone --nproc_per_node=4 \
    scripts/07_domain_parallel_shardtensor/02_shardtensor_conv.py

# 3. Domain parallel training
torchrun --standalone --nproc_per_node=4 \
    scripts/07_domain_parallel_shardtensor/03_domain_parallel_training.py

# 4. Full domain parallel + FSDP hybrid
torchrun --standalone --nproc_per_node=4 \
    scripts/07_domain_parallel_shardtensor/04_domain_parallel_with_fsdp.py
```

**See also:**
- [`scripts/07_domain_parallel_shardtensor/01_why_splitting_fails.py`](../../scripts/07_domain_parallel_shardtensor/01_why_splitting_fails.py) — start here to see the boundary problem
- [`scripts/07_domain_parallel_shardtensor/04_domain_parallel_with_fsdp.py`](../../scripts/07_domain_parallel_shardtensor/04_domain_parallel_with_fsdp.py) — production pattern with FSDP
- [`scripts/07_domain_parallel_shardtensor/README.md`](../../scripts/07_domain_parallel_shardtensor/README.md) — deep dive on domain parallelism

## What's Next?

Now you know all seven strategies. Chapter 11 helps you choose the right
one (or combination) for your specific workload.

**Next:** [Chapter 11 — Choosing a Strategy](11_choosing_a_strategy.md)
