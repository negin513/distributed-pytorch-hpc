# Chapter 8: Sequence Parallelism (SP)
 
In previous chapters, we sharded **parameters** (FSDP), split **layers** across GPUs (Pipeline Parallelism), and divided **weight matrices** within a layer (Tensor Parallelism). Sequence Parallelism tackles the remaining dimension: it shards the **sequence itself** across GPUs.
 
While Sequence Parallelism is conceptually straightforward, its implementation is non-trivial due to the need for communication between GPUs during attention computation. 
SP is essential for training transformer models on long sequences, which is common in LLMs but also becoming increasingly relevant in Earth System Science (ESS) as we move to higher resolution and more complex models.
  
In transformer-based weather and climate models, the input is typically a global atmospheric state tokenized into patches. The sequence length depends on resolution and patch size:
 
| Resolution | Grid Points | Patch Size | Sequence Length |
|:---:|:---:|:---:|:---:|
| 1.0° | 181 × 360 = 65K | 4×4 | ~4K tokens |
| 0.25° | 721 × 1440 = 1M | 4×4 | ~65K tokens |
| 0.25° | 721 × 1440 = 1M | 1×1 | ~1M tokens |
| 0.1° | 1801 × 3600 = 6.5M | 4×4 | ~400K tokens |
 
On top of that, each grid point carries multiple variables (temperature, humidity, wind, geopotential) across multiple pressure levels, further increasing the sequence length. With 10 variables and 20 pressure levels, the effective sequence length multiplies by 200×, making SP even more critical.
 
!!! warning "The quadratic attention problem"
    Self-attention computes an \(N \times N\) attention matrix where \(N\) is the sequence length. At 0.25° resolution with \(1 \times 1\) patches, that's a **1M × 1M** matrix — roughly 4 TB in FP32. This doesn't fit on any single GPU. Even with linear attention variants, the activations alone for long sequences can exceed GPU memory.
 
The memory consumed by **activations** during training scales linearly with sequence length (and quadratically for attention). Unlike parameters and optimizer states (which FSDP handles), activations grow with the input size. Sequence parallelism directly addresses this by splitting the sequence dimension across GPUs.
 
## How Sequence Parallelism Works
 
The core idea: split the input sequence into chunks along the sequence dimension and assign each chunk to a different GPU. Each GPU processes only its portion of the sequence.
 
With 4 GPUs and a sequence of length $N$:
 
```
GPU 0: tokens [0,     N/4)
GPU 1: tokens [N/4,   N/2)
GPU 2: tokens [N/2,   3N/4)
GPU 3: tokens [3N/4,  N)
```
 
The challenge is that **attention requires all tokens to see each other**. So at every attention layer, GPUs must communicate to exchange the key/value information needed for the full attention computation. Different SP strategies handle this communication differently.
 
## SP Strategies
 
### DeepSpeed Ulysses
 
Ulysses partitions the input along the sequence dimension. Before each attention layer, it performs an **all-to-all** collective to redistribute the data so that each GPU holds all tokens for a subset of attention heads. After attention, another all-to-all restores the original partitioning.
 
```
Input:     Each GPU has N/P tokens, all heads
                    ↓
            all-to-all
                    ↓
Attention:  Each GPU has all N tokens, H/P heads
                    ↓
            all-to-all
                    ↓
Output:     Each GPU has N/P tokens, all heads
```
 
```python
# Conceptual Ulysses SP implementation
import torch.distributed as dist
 
def ulysses_attention(q, k, v, sp_group):
    """
    q, k, v: [batch, local_seq_len, num_heads, head_dim]
    After all-to-all: [batch, full_seq_len, local_heads, head_dim]
    """
    # Redistribute: split heads, gather sequence
    q = all_to_all(q, sp_group, scatter_dim=2, gather_dim=1)
    k = all_to_all(k, sp_group, scatter_dim=2, gather_dim=1)
    v = all_to_all(v, sp_group, scatter_dim=2, gather_dim=1)
 
    # Standard attention on full sequence, subset of heads
    out = flash_attention(q, k, v)
 
    # Redistribute back: gather heads, split sequence
    out = all_to_all(out, sp_group, scatter_dim=1, gather_dim=2)
    return out
```
 
!!! tip "Why Ulysses works well on HPC"
    All-to-all is a well-optimized collective in NCCL. On Derecho's 4 A100 GPUs connected via NVLink, the all-to-all overhead is modest. Ulysses is also straightforward to integrate with existing codebases — you wrap the attention call without restructuring the model.
 
### Ring Attention
 
Ring Attention organizes GPUs in a logical ring. Each GPU holds its local chunk of Q (queries) and iteratively receives K/V (key/value) chunks from the next GPU in the ring, computing partial attention at each step.
 
```
Step 0: GPU 0 has Q0, K0, V0 → compute local attention
Step 1: GPU 0 has Q0, K1, V1 → compute partial attention (received from GPU 1)
Step 2: GPU 0 has Q0, K2, V2 → compute partial attention (received from GPU 2)
Step 3: GPU 0 has Q0, K3, V3 → compute partial attention (received from GPU 3)
→ Combine all partial results
```
 
```
Ring topology (4 GPUs):
 
    GPU 0 ──send K,V──→ GPU 1
      ↑                    │
      │                    ↓
    GPU 3 ←──send K,V── GPU 2
```
 
The key advantage: **communication overlaps with compute**. While a GPU computes attention with the current K/V chunk, it simultaneously sends its own K/V to the next GPU and receives the next chunk. With enough compute per step, the communication is fully hidden.
 
```python
# Conceptual Ring Attention
def ring_attention(q_local, k_local, v_local, sp_group):
    """
    Each GPU holds: q_local [batch, N/P, heads, dim]
                     k_local, v_local (same shape)
    """
    num_steps = dist.get_world_size(sp_group)
    k_recv, v_recv = k_local.clone(), v_local.clone()
    running_sum = torch.zeros_like(q_local)
    running_lse = torch.full_like(q_local[..., :1], float('-inf'))
 
    for step in range(num_steps):
        # Async: send current K,V to next GPU, receive from prev
        if step < num_steps - 1:
            send_op = dist.isend(k_recv, dst=next_rank)
            recv_k = torch.empty_like(k_local)
            recv_op = dist.irecv(recv_k, src=prev_rank)
            # (same for V)
 
        # Compute partial attention with current K,V chunk
        attn_out, lse = flash_attention_with_lse(q_local, k_recv, v_recv)
 
        # Online softmax: combine with running result
        running_sum, running_lse = online_softmax_merge(
            running_sum, running_lse, attn_out, lse
        )
 
        if step < num_steps - 1:
            send_op.wait()
            recv_op.wait()
            k_recv = recv_k  # use received chunk next iteration
 
    return running_sum
```
 
!!! note "Online softmax"
    Ring Attention relies on the **online softmax** trick (from FlashAttention) to incrementally combine partial attention results without ever materializing the full $N \times N$ attention matrix. Each step produces a local softmax result, and these are merged using log-sum-exp corrections.
 
### Ulysses vs Ring Attention
 
| | Ulysses | Ring Attention |
|---|---|---|
| **Communication** | 2 × all-to-all per layer | P2P send/recv (ring) |
| **Overlap** | No overlap with compute | Communication overlaps compute |
| **Constraint** | Sequence length divisible by SP degree | Sequence length divisible by SP degree |
| **Constraint** | Number of heads divisible by SP degree | No head constraint |
| **Scaling** | Best for moderate SP degree (≤ 8) | Scales to very large SP degree |
| **Implementation** | Simpler | More complex (online softmax) |
| **Best for** | Within-node SP on Derecho | Cross-node SP at large scale |
 

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
