# Sequence Parallelism (SP)

Sequence parallelism splits the **sequence dimension** across GPUs so that no
single device needs to hold the full sequence length. This enables training on
much longer contexts than a single GPU's memory allows.

There are **three mature approaches**, each with different communication
patterns and trade-offs:

| | Megatron-SP | DeepSpeed-Ulysses | Ring Attention |
|---|---|---|---|
| **Communication** | all-gather + reduce-scatter | all-to-all × 2 | P2P send/recv (ring) |
| **Works standalone?** | No (requires TP) | Yes | Yes |
| **Splits** | Sequence on norms, heads on attn | Heads ↔ sequence | Sequence (KV rotates) |
| **Best for** | TP-heavy LLM training | Moderate sequence lengths | Ultra-long sequences |

---

## 1. Megatron-SP (all-gather / reduce-scatter)

Megatron-style SP is a **natural extension of Tensor Parallelism**. It splits
activations along the sequence dimension during LayerNorm and Dropout (which
operate per-token and don't need cross-token communication), then uses
all-gather to reconstruct the full sequence before attention.

**Key insight**: LayerNorm normalizes over the last dimension (D) independently
per token, so it produces correct results on sequence chunks without any
communication.

```
With Megatron-SP (4 GPUs):

┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│ GPU 0  │  │ GPU 1  │  │ GPU 2  │  │ GPU 3  │
│[B,S/4,D│  │[B,S/4,D│  │[B,S/4,D│  │[B,S/4,D│     ← LayerNorm
│]       │  │]       │  │]       │  │]       │       (local, no comm)
└───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘
    │           │           │           │
    └─────┬─────┴─────┬─────┴─────┬─────┘
      all-gather along S to get full [B, S, D]
          │           │           │
          ▼           ▼           ▼
    ┌──────────┐┌──────────┐┌──────────┐┌──────────┐
    │  GPU 0   ││  GPU 1   ││  GPU 2   ││  GPU 3   │
    │ Attn     ││ Attn     ││ Attn     ││ Attn     │ ← Attention
    │ (D/4)    ││ (D/4)    ││ (D/4)    ││ (D/4)    │   (TP split)
    └────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘
         │           │           │           │
         └─────┬─────┴─────┬─────┴─────┬─────┘
          reduce-scatter along S
              │           │           │
              ▼           ▼           ▼
    ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
    │ GPU 0  │  │ GPU 1  │  │ GPU 2  │  │ GPU 3  │
    │[B,S/4,D│  │[B,S/4,D│  │[B,S/4,D│  │[B,S/4,D│ ← Dropout/FFN Norm
    │]       │  │]       │  │]       │  │]       │   (local, no comm)
    └────────┘  └────────┘  └────────┘  └────────┘
```

**Memory savings**: For each norm/dropout layer, activation memory is reduced
by `P×` (number of GPUs). For a 32-layer model, this can save over 1 GB.

---

## 2. DeepSpeed-Ulysses (all-to-all)

Ulysses-style SP uses **all-to-all** communication to redistribute data between
two layouts: **sequence-sharded** and **head-sharded**. Unlike Megatron-SP, it
works standalone without Tensor Parallelism.

**Key insight**: Multi-head attention is embarrassingly parallel across heads.
By redistributing so each GPU gets all tokens but only a subset of heads,
attention computation is fully local.

```
DeepSpeed-Ulysses (4 GPUs, 8 heads):

Input: each GPU has S/4 tokens, all 8 heads
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│    GPU 0     │  │    GPU 1     │  │    GPU 2     │  │    GPU 3     │
│[B, S/4, 8, D]│  │[B, S/4, 8, D]│  │[B, S/4, 8, D]│  │[B, S/4, 8, D]│
│ tok 0..63    │  │ tok 64..127  │  │ tok 128..191 │  │ tok 192..255 │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │
       └────────┬────────┴────────┬────────┴────────┬────────┘
            all-to-all  (sequence-sharded → head-sharded)
                │                 │                 │
                ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│    GPU 0     │  │    GPU 1     │  │    GPU 2     │  │    GPU 3     │
│[B, S, 2, D] │  │[B, S, 2, D] │  │[B, S, 2, D] │  │[B, S, 2, D] │
│ heads 0,1   │  │ heads 2,3   │  │ heads 4,5   │  │ heads 6,7   │
│ ALL tokens   │  │ ALL tokens   │  │ ALL tokens   │  │ ALL tokens   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │
       ▼                 ▼                 ▼                 ▼
   Attention          Attention          Attention          Attention
   (full seq,         (full seq,         (full seq,         (full seq,
    2 heads)           2 heads)           2 heads)           2 heads)
       │                 │                 │                 │
       └────────┬────────┴────────┬────────┴────────┬────────┘
            all-to-all  (head-sharded → sequence-sharded)
                │                 │                 │
                ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│    GPU 0     │  │    GPU 1     │  │    GPU 2     │  │    GPU 3     │
│[B, S/4, 8, D]│  │[B, S/4, 8, D]│  │[B, S/4, 8, D]│  │[B, S/4, 8, D]│
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

---

## 3. Ring Attention (P2P ring)

Ring Attention keeps each GPU's query (Q) **local** and rotates key-value (KV)
blocks around a ring using point-to-point send/recv. At each step, partial
attention is computed and accumulated using **online softmax** (log-sum-exp
correction). After all rotations, each GPU has the correct full attention output.

**Key insight**: No GPU ever holds the full sequence. The attention score matrix
`[S, S]` is never materialized — only `[S/P, S/P]` blocks are computed at a time.
This makes Ring Attention ideal for ultra-long contexts.

```
Ring Attention (4 GPUs):

Q stays local; KV rotates around the ring

Step 0: Each GPU computes attention with its own KV
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  GPU 0  │    │  GPU 1  │    │  GPU 2  │    │  GPU 3  │
│ Q₀×K₀ᵀ │    │ Q₁×K₁ᵀ │    │ Q₂×K₂ᵀ │    │ Q₃×K₃ᵀ │
└─────────┘    └─────────┘    └─────────┘    └─────────┘

Step 1: KV rotates one position (GPU i sends to GPU i+1)
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  GPU 0  │    │  GPU 1  │    │  GPU 2  │    │  GPU 3  │
│ Q₀×K₃ᵀ │←──│ Q₁×K₀ᵀ │←──│ Q₂×K₁ᵀ │←──│ Q₃×K₂ᵀ │←─┐
└─────────┘    └─────────┘    └─────────┘    └─────────┘  │
     └────────────────────────────────────────────────────┘

Step 2: KV rotates again
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  GPU 0  │    │  GPU 1  │    │  GPU 2  │    │  GPU 3  │
│ Q₀×K₂ᵀ │    │ Q₁×K₃ᵀ │    │ Q₂×K₀ᵀ │    │ Q₃×K₁ᵀ │
└─────────┘    └─────────┘    └─────────┘    └─────────┘

Step 3: KV rotates again → each GPU has seen all KV blocks
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  GPU 0  │    │  GPU 1  │    │  GPU 2  │    │  GPU 3  │
│ Q₀×K₁ᵀ │    │ Q₁×K₂ᵀ │    │ Q₂×K₃ᵀ │    │ Q₃×K₀ᵀ │
└─────────┘    └─────────┘    └─────────┘    └─────────┘

Online softmax accumulates partial results at each step:
  out = α·out_prev + β·out_block   (α, β from log-sum-exp correction)

After 4 steps: each GPU has correct full attention output
for its S/4 query tokens, without ever holding the full sequence.
```

---

## Comparison

| | Megatron-SP | DeepSpeed-Ulysses | Ring Attention |
|---|---|---|---|
| **Communication** | all-gather + reduce-scatter | all-to-all × 2 | P2P send/recv (ring) |
| **Volume per layer** | O(B·S·D) | O(B·S·D) | O(B·S·D) but pipelined |
| **Works standalone?** | No (requires TP) | Yes | Yes |
| **Constraint** | D divisible by P | H divisible by P | S divisible by P |
| **Attention type** | TP-split (D/P heads per GPU) | Full seq, H/P heads | Full heads, S/P queries |
| **Memory** | Saves on norms/dropout | Same as standard per head | Never holds full S×S scores |
| **Best for** | TP-heavy LLM training | Moderate seq, many heads | Ultra-long contexts (100K+) |
| **PyTorch API** | `SequenceParallel()` | Manual all-to-all | Manual P2P |

---

## Files in this Directory

| Script | GPUs | Approach | What it demonstrates |
|--------|------|----------|---------------------|
| `01_basic_sequence_parallel.py` | 4 | Megatron | Splitting sequence dim, LayerNorm locality, all-gather/reduce-scatter |
| `02_sp_transformer_layer.py` | 4 | Megatron | Applying SP to a transformer block with TP using PyTorch APIs |
| `03_sp_training.py` | 4 | Megatron | Full training with SP+TP, memory comparison, throughput measurement |
| `04_ulysses_sequence_parallel.py` | 4 | Ulysses | All-to-all redistribution between sequence-sharded and head-sharded layouts |
| `05_ring_attention_concept.py` | 4 | Ring | P2P KV rotation with online softmax accumulation, memory analysis |

---

## Running the Examples

### With mpiexec (recommended on Derecho)

```bash
# Megatron-SP
mpiexec -n 4 --ppn 4 --cpu-bind none python 01_basic_sequence_parallel.py
mpiexec -n 4 --ppn 4 --cpu-bind none python 02_sp_transformer_layer.py
mpiexec -n 4 --ppn 4 --cpu-bind none python 03_sp_training.py

# DeepSpeed-Ulysses
mpiexec -n 4 --ppn 4 --cpu-bind none python 04_ulysses_sequence_parallel.py

# Ring Attention
mpiexec -n 4 --ppn 4 --cpu-bind none python 05_ring_attention_concept.py
```

### With torchrun (single node, 4 GPUs)

```bash
torchrun --standalone --nproc_per_node=4 01_basic_sequence_parallel.py
torchrun --standalone --nproc_per_node=4 02_sp_transformer_layer.py
torchrun --standalone --nproc_per_node=4 03_sp_training.py
torchrun --standalone --nproc_per_node=4 04_ulysses_sequence_parallel.py
torchrun --standalone --nproc_per_node=4 05_ring_attention_concept.py
```

### Multi-node with mpiexec (2 nodes × 4 GPUs)

```bash
export NCCL_SOCKET_IFNAME=hsn
export NCCL_IB_DISABLE=1
export NCCL_CROSS_NIC=1
mpiexec -n 8 --ppn 4 --cpu-bind none python 03_sp_training.py --seq-len 2048
```

### Via PBS on Derecho

```bash
qsub run_sequence_parallel.sh
```

---

## When to Use Each Approach

**Choose Megatron-SP when:**
- You are already using Tensor Parallelism
- You want to reduce activation memory on norms and dropout
- Training large transformers (LLMs) with standard sequence lengths

**Choose DeepSpeed-Ulysses when:**
- You want sequence parallelism without TP
- Your model has enough attention heads (H ≥ P)
- Moderate sequence lengths where all-to-all is efficient

**Choose Ring Attention when:**
- Sequences are extremely long (100K+ tokens)
- Memory is the bottleneck (can't fit S×S attention scores)
- You want to scale sequence length independently of model size
- Willing to trade some latency for memory savings

---

## Prerequisites

- PyTorch >= 2.3
- 4 GPUs minimum

## References

- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) — Megatron-SP
- [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509) — Ulysses
- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) — Ring Attention
- [PyTorch SequenceParallel API](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
