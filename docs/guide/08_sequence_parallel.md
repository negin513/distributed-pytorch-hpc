# Chapter 8: Sequence Parallel (SP)

Attention memory scales as O(n^2) with sequence length. When context
windows reach tens or hundreds of thousands of tokens, even the
activations for a single layer can exceed GPU memory. Sequence
Parallelism splits the sequence dimension across GPUs.

## The Problem: Long Sequences

Self-attention computes an [n × n] attention matrix, where n is the
sequence length:

```
Sequence length    Attention matrix    Memory (FP16)
─────────────────────────────────────────────────────
    2,048           2K × 2K              8 MB
    8,192           8K × 8K            128 MB
   32,768          32K × 32K            2 GB
  131,072         128K × 128K          32 GB
  524,288         512K × 512K         512 GB
```

With multiple layers and attention heads, the activation memory for long
sequences dominates everything else.

## Three Approaches

There are three main approaches to sequence parallelism, each with
different tradeoffs:

### 1. Megatron-SP: TP Extension for LayerNorm and Dropout

Megatron-SP works alongside Tensor Parallelism. It splits the sequence
dimension for operations that TP doesn't parallelize — LayerNorm,
Dropout, and residual connections:

```
Standard TP (without SP):
  LayerNorm runs on full sequence (replicated) → wasteful
  Attention/FFN split across GPUs by TP        → efficient

Megatron-SP (with TP):
  LayerNorm split by sequence dimension         → efficient
  ── all-gather ──
  Attention/FFN split across GPUs by TP         → efficient
  ── reduce-scatter ──
  LayerNorm split by sequence dimension         → efficient
```

```
Forward pass (2 GPUs, TP + Megatron-SP):

GPU 0: LayerNorm(seq[0:n/2]) ──┐
GPU 1: LayerNorm(seq[n/2:n]) ──┤── all-gather ──► full sequence
                                │
GPU 0: Attn(full_seq, heads[0:h/2]) ──┐
GPU 1: Attn(full_seq, heads[h/2:h]) ──┤── reduce-scatter
                                       │
GPU 0: LayerNorm(seq[0:n/2])
GPU 1: LayerNorm(seq[n/2:n])
```

**Key property:** Megatron-SP requires TP. The all-gather/reduce-scatter
replace the all-reduce that TP would normally use, so there is **no extra
communication cost** — just a different arrangement.

### 2. DeepSpeed-Ulysses: All-to-All Redistribution

Ulysses splits the sequence evenly and uses all-to-all to redistribute
between sequence-sharded and head-sharded layouts:

```
Step 1: Each GPU has part of sequence, all heads
  GPU 0: seq[0:n/4], heads[0:h]
  GPU 1: seq[n/4:n/2], heads[0:h]
  GPU 2: seq[n/2:3n/4], heads[0:h]
  GPU 3: seq[3n/4:n], heads[0:h]

Step 2: all-to-all → each GPU has full sequence, part of heads
  GPU 0: seq[0:n], heads[0:h/4]
  GPU 1: seq[0:n], heads[h/4:h/2]
  GPU 2: seq[0:n], heads[h/2:3h/4]
  GPU 3: seq[0:n], heads[3h/4:h]

Step 3: Each GPU computes attention on its heads (full sequence)

Step 4: all-to-all → back to sequence-sharded layout
```

**Key property:** Does **not** require TP. Works independently.
Communication is two all-to-all operations per attention layer.

### 3. Ring Attention: P2P KV Rotation

Ring Attention computes attention in chunks, rotating KV blocks around a
ring of GPUs:

```
Ring of 4 GPUs:

     GPU 0 ──► GPU 1
       ▲         │
       │         ▼
     GPU 3 ◄── GPU 2

Each GPU:
  1. Has its own Q chunk (stays put)
  2. Receives a KV chunk from its neighbor
  3. Computes partial attention with that KV chunk
  4. Passes the KV chunk to the next GPU
  5. Repeat until all KV chunks have visited all GPUs

Uses online softmax (log-sum-exp trick) to combine partial results.
```

**Key property:** Theoretically supports infinite context length —
each GPU only needs memory for its chunk. Communication is overlapped
with computation via async send/recv.

## Comparison

| Approach | Communication | TP Required? | Max Efficiency |
|----------|--------------|-------------|----------------|
| **Megatron-SP** | All-gather + reduce-scatter | Yes | High (replaces TP comm) |
| **Ulysses** | 2× all-to-all per layer | No | High (with enough heads) |
| **Ring Attention** | P2P send/recv (overlapped) | No | Medium (limited overlap) |

### When to Use Each

- **Megatron-SP:** You're already using TP and want to save activation
  memory from LayerNorm/Dropout. No extra communication cost.
- **Ulysses:** You want SP without TP, and your model has enough
  attention heads to shard (num_heads >= SP degree).
- **Ring Attention:** You need very long contexts (100K+ tokens) and
  can tolerate some overhead for the flexibility.

## Running the Examples

The scripts progress from basic concepts to full implementations:

```bash
# 1. Basic Megatron-SP mechanics
torchrun --standalone --nproc_per_node=4 \
    scripts/05_sequence_parallel_sp/01_basic_sequence_parallel.py

# 2. SP applied to a transformer layer with TP
torchrun --standalone --nproc_per_node=4 \
    scripts/05_sequence_parallel_sp/02_sp_transformer_layer.py

# 3. Full training with SP+TP
torchrun --standalone --nproc_per_node=4 \
    scripts/05_sequence_parallel_sp/03_sp_training.py

# 4. Ulysses (all-to-all) approach
torchrun --standalone --nproc_per_node=4 \
    scripts/05_sequence_parallel_sp/04_ulysses_sequence_parallel.py

# 5. Ring Attention concept
torchrun --standalone --nproc_per_node=4 \
    scripts/05_sequence_parallel_sp/05_ring_attention_concept.py
```

**See also:**
- [`scripts/05_sequence_parallel_sp/`](../../scripts/05_sequence_parallel_sp/) — all five SP examples
- [`scripts/05_sequence_parallel_sp/README.md`](../../scripts/05_sequence_parallel_sp/README.md) — deep dive on SP

## What's Next?

For models larger than ~7B parameters, you typically need to combine
multiple strategies. Chapter 9 covers the most common combination:
TP within nodes + FSDP across nodes.

**Next:** [Chapter 9 — Hybrid Parallelism](09_hybrid_parallelism.md)
