# Chapter 6: Tensor Parallel (TP)

Tensor Parallelism splits individual **weight matrices** across GPUs. While
FSDP shards whole parameters and reconstructs them before use, TP keeps
each GPU's shard in place and computes partial results that are combined
with a collective operation.

## When You Need TP

Consider a large transformer's feed-forward layer:

```
FFN weight: [hidden_size × 4*hidden_size]

For LLaMA-70B: [8192 × 32768] = 268M parameters = 1 GB in FP32

A single matmul on this weight requires the full matrix in GPU memory.
FSDP would all-gather it before use — but what if even temporarily
holding the full weight doesn't fit?
```

TP solves this by never assembling the full matrix. Each GPU permanently
holds a slice and computes a partial result.

## Column-Parallel Linear

Split the weight matrix by **columns**. Each GPU has half the output
features and computes its portion independently:

```
Full operation:  Y = XA    where A is [K × N]

Split A into columns:  A = [A₁ | A₂]

GPU 0:  Y₁ = X · A₁     (computes left half of output)
GPU 1:  Y₂ = X · A₂     (computes right half of output)

Result: Y = [Y₁ | Y₂]   (concatenate)
```

```
         X (input)                 A (weights)
     ┌──────────┐          ┌──────────┬──────────┐
     │          │    ×     │    A₁    │    A₂    │
     │  [B × K] │          │  [K×N/2] │  [K×N/2] │
     └──────────┘          └──────────┴──────────┘
                                GPU 0      GPU 1
                                  │          │
                                  ▼          ▼
                              Y₁=[B×N/2]  Y₂=[B×N/2]
```

No communication needed for the forward pass — each GPU has all of X.

## Row-Parallel Linear

Split the weight matrix by **rows**. Each GPU has half the input features
and computes a partial sum:

```
Full operation:  Y = XA    where A is [K × N]

Split A into rows:  A = [A₁]    Split X into columns:  X = [X₁ | X₂]
                    [A₂]

GPU 0:  Y₀ = X₁ · A₁     (partial result)
GPU 1:  Y₁ = X₂ · A₂     (partial result)

Result: Y = Y₀ + Y₁       (all-reduce to sum)
```

Row-parallel **requires an all-reduce** to combine partial sums.

## The Column → Row Pattern

The power of TP comes from chaining column-parallel and row-parallel
layers. In a transformer FFN:

```
FFN:  Y = dropout(GeLU(X · W₁) · W₂)

With TP:
  W₁ = column-parallel  →  each GPU gets half of GeLU output
  W₂ = row-parallel     →  all-reduce combines partial sums

         X
         │
    ┌────┴────┐
    ▼         ▼
  X·W₁ᵃ    X·W₁ᵇ      ← column-parallel (no comm)
    │         │
  GeLU      GeLU
    │         │
  ·W₂ᵃ     ·W₂ᵇ       ← row-parallel
    │         │
    └────┬────┘
     all-reduce          ← one communication per FFN
         │
         Y
```

The column-parallel output naturally feeds the row-parallel input, so
**communication cancels out** — you only need one all-reduce per FFN
block instead of two.

## PyTorch TP API

PyTorch provides `DeviceMesh` and `parallelize_module` for TP:

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

# Create a 1D mesh for TP across 4 GPUs
tp_mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("tp",))

# Parallelize specific layers
model = parallelize_module(
    model,
    tp_mesh,
    {
        "ffn.w1": ColwiseParallel(),   # split columns
        "ffn.w2": RowwiseParallel(),   # split rows
        "attn.qkv": ColwiseParallel(), # split Q, K, V projections
        "attn.out": RowwiseParallel(), # combine attention output
    },
)
```

## TP Degree on Derecho

TP requires frequent all-reduces between GPUs. On systems with NVLink
(600+ GB/s), TP can scale to 8 GPUs. On Derecho with PCIe (~25 GB/s),
keep TP within a single node:

```
Derecho: 4 GPUs per node, PCIe Gen4

Recommended: TP degree = 4 (one full node)

Going beyond TP=4 puts TP communication on the slower Slingshot
fabric, which hurts throughput.
```

For larger models, combine TP with FSDP across nodes (Chapter 9).

## 1D vs 2D Tensor Parallelism

**1D TP** splits weight matrices along one dimension (columns or rows),
as shown above.

**2D TP** splits along both dimensions using a 2D GPU grid. This reduces
communication volume but requires more GPUs. With 4 GPUs in a 2×2 grid:

```
Weight matrix A [K × N]:

     GPU (0,0)     GPU (0,1)
   ┌───────────┬───────────┐
   │ A[0:K/2,  │ A[0:K/2,  │
   │   0:N/2]  │   N/2:N]  │
   ├───────────┼───────────┤
   │ A[K/2:K,  │ A[K/2:K,  │
   │   0:N/2]  │   N/2:N]  │
   └───────────┴───────────┘
     GPU (1,0)     GPU (1,1)

Communication: reduce-scatter along rows, all-gather along columns
```

2D TP reduces the per-GPU communication from O(N) to O(√N) but adds
complexity. See the scripts for a working example.

## Running the Examples

The TP scripts are progressive — start with 01 and work through:

```bash
# Start here: basic TP concepts
torchrun --standalone --nproc_per_node=4 \
    scripts/03_tensor_parallel_tp/01_basic_tensor_parallel.py

# DeviceMesh for organizing GPUs
torchrun --standalone --nproc_per_node=4 \
    scripts/03_tensor_parallel_tp/02_device_mesh_example.py

# 2D tensor parallelism
torchrun --standalone --nproc_per_node=4 \
    scripts/03_tensor_parallel_tp/03_2d_tensor_parallel.py

# Advanced patterns
torchrun --standalone --nproc_per_node=4 \
    scripts/03_tensor_parallel_tp/04_advanced_tp_example.py
```

**See also:**
- [`scripts/03_tensor_parallel_tp/`](../../scripts/03_tensor_parallel_tp/) — progressive TP examples (01 → 04)
- [`scripts/03_tensor_parallel_tp/README.md`](../../scripts/03_tensor_parallel_tp/README.md) — deep dive on TP

## What's Next?

TP splits layers horizontally (by weight matrix). Pipeline Parallelism
splits the model vertically — assigning different layers to different
GPUs.

**Next:** [Chapter 7 — Pipeline Parallel](07_pipeline_parallel.md)
