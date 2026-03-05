# Parallelism Strategy Decision Guide

Use this guide to choose the right distributed training strategy for your
workload on Derecho.

## Decision Flowchart

```
                    Does your model fit on 1 GPU?
                    ┌───────────┴───────────┐
                   YES                      NO
                    │                        │
              Use DDP (01_*)          What doesn't fit?
              (simplest, best          ┌─────┴──────┐
               scaling for          Too many      Too many
               data parallel)       parameters    activations
                                       │              │
                                 Is it deep        Is input
                                 (many layers)?    spatially large?
                                 ┌────┴────┐      ┌────┴────┐
                                YES       NO     YES       NO
                                 │         │      │         │
                              Use PP    Use TP  Domain    Use FSDP
                             (04_*)   (03_*)   Parallel    (02_*)
                                │         │    (07_*)
                                │         │
                          ┌─────┴─────────┴─────┐
                          │  Still doesn't fit?  │
                          │  Combine strategies: │
                          │                      │
                          │  TP + FSDP (06_*)    │
                          │  TP + PP + FSDP      │
                          │  TP + SP (05_*)      │
                          └──────────────────────┘
```

## Strategy Comparison Table

| Strategy | What's Split | Communication | Memory Savings | Best For |
|----------|-------------|---------------|----------------|----------|
| **DDP** | Data (batches) | Gradient all-reduce | None (full replica) | Most workloads |
| **FSDP** | Params + grads + optimizer | All-gather + reduce-scatter | High (params) | Large models |
| **TP** | Weight matrices | All-reduce on activations | Medium (weights) | Wide layers |
| **PP** | Model layers | Send/recv between stages | High (layers) | Very deep models |
| **SP** | Sequence dimension | All-gather + reduce-scatter | Medium (activations) | Long sequences |
| **Domain** | Spatial data | Halo exchange (P2P) | High (spatial) | High-res spatial data |

## Quick Reference by Use Case

### "I want to train faster on more GPUs"
→ **DDP** (`01_data_parallel_ddp/`)

Your model fits on 1 GPU, you just want to process more data per step.
Each GPU gets a different batch, gradients are averaged. Linear speedup.

### "My model doesn't fit on 1 GPU"
→ **FSDP** (`02_fully_sharded_fsdp/`)

FSDP shards parameters, gradients, and optimizer states across GPUs.
A 10B parameter model that needs 80 GB can train on 4× 40 GB A100s.

### "I have very large linear layers (e.g., LLM)"
→ **TP** (`03_tensor_parallel_tp/`)

Split large weight matrices across GPUs. Good for transformers where
individual layers (attention, FFN) are too large.

### "My model has 100+ layers"
→ **PP** (`04_pipeline_parallel_pp/`)

Assign different layers to different GPUs. Use micro-batching to keep
GPUs busy. Often combined with TP for very large models.

### "Long sequences are blowing up my memory"
→ **SP** (`05_sequence_parallel_sp/`)

Split the sequence dimension for LayerNorm/Dropout. Saves activation
memory. Usually combined with TP (SP is an extension of TP).

### "My training already uses TP + FSDP"
→ **Hybrid** (`06_hybrid_parallelism/`)

TP within nodes (fast PCIe), FSDP across nodes. The standard approach
for training large language models at scale.

### "My input images/grids are too large for 1 GPU"
→ **Domain Parallel** (`07_domain_parallel_shardtensor/`)

Split spatial dimensions across GPUs. Critical for weather/climate models,
medical imaging, and CFD with high-resolution grids.

## Derecho-Specific Recommendations

```
Derecho: 82 nodes × 4 A100 (40 GB) — PCIe (no NVLink)

Recommended configurations:
─────────────────────────────────────────────────────
 GPUs    Nodes    Strategy              Notes
─────────────────────────────────────────────────────
  4       1       DDP                   Start here
  4       1       FSDP                  If model > 40 GB
  4       1       TP (tp=4)             If layers > 40 GB
  8       2       DDP                   Scale data parallel
  8       2       TP=4 + FSDP=2         TP within node, FSDP across
 16       4       TP=4 + FSDP=4         Standard LLM training
 32       8       TP=4 + FSDP=8         Large-scale training
 328     82       TP=4 + FSDP=82        Full cluster
─────────────────────────────────────────────────────

Note: Since Derecho uses PCIe (not NVLink), TP communication
cost is higher than on DGX systems. Keep TP degree ≤ 4
(within a single node).
```
