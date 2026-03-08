# Chapter 9: Hybrid Parallelism

When models exceed ~7B parameters, no single strategy is enough. Hybrid
parallelism combines TP + FSDP (and optionally PP and SP) to handle the
largest models. This is how large language models are trained in practice.

## Why Combine Strategies?

Each strategy has strengths and limitations:

| Strategy | Good at | Limited by |
|----------|---------|-----------|
| DDP | Scaling data throughput | Model must fit on 1 GPU |
| FSDP | Sharding optimizer state | All-gather reconstructs full layers |
| TP | Splitting large layers | Needs fast interconnect |
| PP | Deep models | Pipeline bubbles |

For a 13B model on Derecho (4× 40 GB A100 per node):
- DDP: model doesn't fit on one GPU
- FSDP alone: works with 2+ nodes, but all-gathers are large
- TP alone: limited to 4 GPUs (one node)
- **TP + FSDP:** TP splits layers within a node, FSDP shards across nodes

## The 2D Device Mesh

Hybrid parallelism organizes GPUs into a 2D grid. One dimension is for
TP (fast, intra-node), the other for FSDP (across nodes):

```
2 nodes × 4 GPUs = 8 GPUs total

                    TP dimension (intra-node)
                ┌──────────────────────────────┐
                │  GPU 0    GPU 1    GPU 2    GPU 3  │  Node 0
FSDP dimension  │  (0,0)   (0,1)   (0,2)   (0,3)  │
(inter-node)    ├──────────────────────────────┤
                │  GPU 4    GPU 5    GPU 6    GPU 7  │  Node 1
                │  (1,0)   (1,1)   (1,2)   (1,3)  │
                └──────────────────────────────┘

TP groups (communicate via NVLink, fast):
  [GPU 0, 1, 2, 3]   ← Node 0
  [GPU 4, 5, 6, 7]   ← Node 1

FSDP groups (communicate via Slingshot, still fast on Derecho):
  [GPU 0, 4]   [GPU 1, 5]   [GPU 2, 6]   [GPU 3, 7]
```

In PyTorch:

```python
from torch.distributed.device_mesh import init_device_mesh

# 2D mesh: FSDP across 2 nodes, TP across 4 GPUs per node
mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("fsdp", "tp"))

tp_mesh = mesh["tp"]      # for parallelize_module
fsdp_mesh = mesh["fsdp"]  # for FSDP wrapping
```

## The Pattern: TP First, Then FSDP

The standard approach for each transformer block:

1. **Apply TP** to the attention and FFN layers (column/row parallel)
2. **Wrap with FSDP** so parameters are sharded across the FSDP dimension

```python
from torch.distributed.tensor.parallel import (
    parallelize_module, ColwiseParallel, RowwiseParallel,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

for block in model.transformer_blocks:
    # Step 1: TP within node
    parallelize_module(block, tp_mesh, {
        "attn.qkv_proj": ColwiseParallel(),
        "attn.out_proj": RowwiseParallel(),
        "ffn.w1":        ColwiseParallel(),
        "ffn.w2":        RowwiseParallel(),
    })

# Step 2: FSDP across nodes
model = FSDP(model, device_mesh=fsdp_mesh, ...)
```

## Data Flow Through a Hybrid Block

```
Input (batch on this FSDP rank)
  │
  ├─► LayerNorm (local, optionally SP)
  │
  ├─► Attention
  │     Q, K, V projections: ColwiseParallel (TP splits heads)
  │     Attention compute: each GPU handles its heads
  │     Output projection: RowwiseParallel (TP all-reduce)
  │
  ├─► Residual connection
  │
  ├─► LayerNorm (local, optionally SP)
  │
  ├─► FFN
  │     W1: ColwiseParallel (TP splits hidden dim)
  │     GeLU (local)
  │     W2: RowwiseParallel (TP all-reduce)
  │
  ├─► Residual connection
  │
  └─► Output

Communication per block:
  - TP: 2 all-reduces (attention + FFN)
  - FSDP: all-gather params before, reduce-scatter grads after
```

## Sizing Guide

How to choose TP and FSDP dimensions for different model sizes on Derecho:

```
Derecho: 82 nodes × 4 A100 (40 GB each)

Model Size    TP    FSDP    Total GPUs    Nodes
──────────────────────────────────────────────────
 < 3B          1      4          4          1       ← FSDP only
  7B           4      2          8          2       ← TP within node
 13B           4      4         16          4
 30B           4      8         32          8
 70B           4     20         80         20
──────────────────────────────────────────────────

Rule of thumb:
  - TP = 4 (always one full node on Derecho)
  - FSDP = total_nodes
  - Total GPUs = TP × FSDP
```

For models smaller than ~3B, FSDP alone is usually sufficient and simpler.

## Adding PP and SP

For the very largest models, you can add Pipeline Parallelism and
Sequence Parallelism to the mix, creating a 3D or 4D mesh:

```
3D: TP + PP + FSDP (e.g., 70B+ models)
  - TP=4 within node
  - PP=4 across 4 nodes (one pipeline stage per node)
  - FSDP=N across remaining nodes

4D: TP + SP + PP + FSDP (e.g., 100B+ with long context)
  - SP shares the TP dimension (Megatron-SP)
  - PP for depth
  - FSDP for the rest
```

These configurations are beyond what's practical on Derecho's 82 GPU
nodes but are standard for large-scale LLM training.

## Running the Examples

```bash
# FSDP + TP hybrid on 8 GPUs (2 nodes)
# Use the PBS script for multi-node:
qsub scripts/06_hybrid_parallelism/run_hybrid.sh

# Or on a single node with tp=4, fsdp=1 (for testing):
torchrun --standalone --nproc_per_node=4 \
    scripts/06_hybrid_parallelism/01_fsdp_tp_hybrid.py
```

**See also:**
- [`scripts/06_hybrid_parallelism/01_fsdp_tp_hybrid.py`](../../scripts/06_hybrid_parallelism/01_fsdp_tp_hybrid.py) — complete TP+FSDP example
- [`scripts/06_hybrid_parallelism/llama2_model.py`](../../scripts/06_hybrid_parallelism/llama2_model.py) — LLaMA 2 model optimized for hybrid parallelism
- [`scripts/06_hybrid_parallelism/README.md`](../../scripts/06_hybrid_parallelism/README.md) — deep dive on hybrid parallelism

## What's Next?

All strategies so far split the model or the batch. Domain Parallelism
is different — it splits the **spatial input data** itself. This is
essential for scientific computing workloads with high-resolution grids.

**Next:** [Chapter 10 — Domain Parallel](10_domain_parallel.md)
