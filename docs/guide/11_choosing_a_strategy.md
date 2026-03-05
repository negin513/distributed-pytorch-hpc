# Chapter 11: Choosing a Strategy

You now know seven distributed training strategies. This chapter helps you
pick the right one — or the right combination — for your workload.

## Decision Flowchart

```
                     Does your model fit on 1 GPU?
                     ┌───────────┴───────────┐
                    YES                      NO
                     │                        │
               ┌─────┴──────┐          What doesn't fit?
               │  Use DDP   │          ┌──────┬──────────┐
               │  (Ch 4)    │        Params  Activations  Input
               │            │          │         │          │
               │  Fastest,  │          │         │     Is it spatial
               │  simplest  │          │         │     (grid/image)?
               └────────────┘          │         │     ┌────┴────┐
                                       │         │    YES       NO
                                       │         │     │         │
                                  Is it deep    Use   Domain    Use SP
                                  (many layers)? FSDP  Parallel  (Ch 8)
                                  ┌────┴────┐  (Ch 5)  (Ch 10)
                                 YES       NO
                                  │         │
                               Use PP    Use TP
                               (Ch 7)   (Ch 6)
                                  │         │
                            ┌─────┴─────────┴──────┐
                            │  Still doesn't fit?   │
                            │  Combine strategies:  │
                            │                       │
                            │  TP + FSDP (Ch 9)     │
                            │  TP + PP + FSDP       │
                            │  TP + SP (Ch 8)       │
                            └───────────────────────┘
```

## Quick Reference: "I Want To..."

| Goal | Strategy | Chapter |
|------|----------|---------|
| Train faster on more GPUs | DDP | [4](04_data_parallel_ddp.md) |
| Fit a model that's 1-3× too large | FSDP | [5](05_fully_sharded_fsdp.md) |
| Fit a model with huge linear layers | TP | [6](06_tensor_parallel.md) |
| Fit a model with 100+ layers | PP | [7](07_pipeline_parallel.md) |
| Handle very long sequences | SP | [8](08_sequence_parallel.md) |
| Train a 7B+ parameter LLM | TP + FSDP | [9](09_hybrid_parallelism.md) |
| Process high-resolution spatial data | Domain Parallel | [10](10_domain_parallel.md) |

## Concrete Scenarios

### Scenario 1: Fine-tuning a vision model

**Model:** ResNet-50 (25M params, ~400 MB with optimizer)
**Data:** ImageNet (1.2M images)
**Goal:** Train faster

**Recommendation:** DDP on 4-16 GPUs. The model easily fits on one GPU.
DDP gives near-linear scaling. Start with 1 node (4 GPUs), scale to 4
nodes if you need more speed.

### Scenario 2: Pre-training a 7B LLM from scratch

**Model:** LLaMA-7B architecture (7B params, ~112 GB with Adam FP32)
**Data:** Large text corpus
**Goal:** Train the model

**Recommendation:** TP=4 + FSDP=N. Use TP within each node (4 GPUs) to
split the large attention and FFN layers. Use FSDP across nodes to shard
the optimizer state. With BF16 mixed precision and 4 nodes (16 GPUs),
this fits comfortably.

### Scenario 3: Global weather prediction at 0.25 degrees

**Model:** U-Net variant (50M params, fits on 1 GPU)
**Data:** 1440×720 grid, 100+ channels
**Goal:** Fit the forward pass in memory

**Recommendation:** Domain Parallel (4 GPUs). Split the grid into 4 tiles,
use halo exchange for convolution boundaries. The model is small but the
input is huge — domain parallel directly addresses this. Add FSDP if the
model grows.

### Scenario 4: Long-context document understanding

**Model:** Transformer (1B params)
**Data:** 128K token documents
**Goal:** Handle long sequences without running out of memory

**Recommendation:** SP + TP. Use Megatron-SP or Ulysses to split the
sequence dimension. Combine with TP to split the attention heads. With
4 GPUs, each GPU handles 32K tokens — much more manageable.

### Scenario 5: Very deep diffusion model

**Model:** 200-layer diffusion model (each layer small, total ~5B params)
**Data:** High-resolution images
**Goal:** Train the deep model

**Recommendation:** PP + FSDP. Pipeline parallelism splits the 200 layers
across stages (e.g., 50 layers per GPU with 4 stages). FSDP shards the
optimizer across additional GPUs. Use 16+ micro-batches to keep pipeline
bubbles small.

## Derecho Configuration Reference

```
NCAR Derecho: 82 nodes × 4 A100 (40 GB), PCIe, Slingshot 11

 GPUs    Nodes    Strategy              Config
─────────────────────────────────────────────────────
  4       1       DDP                   Start here for small models
  4       1       FSDP                  Model 40-160 GB
  4       1       TP (degree=4)         Individual layers > 40 GB
  8       2       DDP                   More data throughput
  8       2       TP=4 + FSDP=2         Model 100-300 GB
 16       4       TP=4 + FSDP=4         7B-13B models
 32       8       TP=4 + FSDP=8         13B-30B models
 80      20       TP=4 + FSDP=20        70B models
328      82       TP=4 + FSDP=82        Full cluster
─────────────────────────────────────────────────────

Key constraint: TP degree ≤ 4 (PCIe, not NVLink).
Always keep TP within a single node.
```

## Common Mistakes

### Over-engineering for small models

If your model fits on one GPU, use DDP. Don't add FSDP "just in case" —
it adds communication overhead with no memory benefit.

### TP across nodes

On Derecho, TP communication over Slingshot is much slower than NVLink
systems expect. Keep TP=4 (one node) and use FSDP for cross-node
sharding.

### Too few micro-batches with PP

Pipeline bubble = (stages - 1) / micro-batches. With 4 stages and 4
micro-batches, 75% of time is bubble. Use at least 4× stages.

### Ignoring activation memory

FSDP shards parameters and optimizer state but not activations. For very
long sequences or high-resolution spatial data, you still need SP or
domain parallelism for activation memory.

## What's Next?

You know *what* to use. Chapter 12 covers the *how* — PBS job scripts,
NCCL tuning, and debugging on Derecho.

**Next:** [Chapter 12 — HPC Operations](12_hpc_operations.md)

---

**See also:** [Strategy Decision Guide](strategy_decision_guide.md) — the quick-reference version of this chapter.
