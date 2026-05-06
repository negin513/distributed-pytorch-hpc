#!/usr/bin/env python3
"""
04_ulysses_sequence_parallel.py — DeepSpeed-Ulysses Sequence Parallelism

Demonstrates all-to-all based sequence parallelism (DeepSpeed-Ulysses style):
    1. Each GPU starts with [B, S/P, H, D] — a chunk of the sequence
    2. all-to-all redistributes to [B, S, H/P, D] — full sequence, subset of heads
    3. Each GPU computes attention on FULL sequence for its H/P heads
    4. all-to-all redistributes back to [B, S/P, H, D]

Unlike Megatron-SP (which requires TP), Ulysses works standalone — it only needs
the number of heads to be divisible by the number of GPUs.

Communication pattern:
    Megatron-SP:  all-gather + reduce-scatter (tied to TP)
    Ulysses:      all-to-all + all-to-all     (standalone)

Run with mpiexec:
    mpiexec -n 4 --ppn 4 --cpu-bind none python 04_ulysses_sequence_parallel.py

Run with torchrun (single node):
    torchrun --standalone --nproc_per_node=4 04_ulysses_sequence_parallel.py

Run with mpiexec (multi-node, 2 nodes x 4 GPUs):
    mpiexec -n 8 --ppn 4 --cpu-bind none python 04_ulysses_sequence_parallel.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
import torch.distributed as dist
from utils.distributed import init_distributed, cleanup_distributed


def all_to_all_seq_to_head(x, world_size):
    """Redistribute [B, S/P, H, D] → [B, S, H/P, D] via all-to-all.

    Split the head dimension into P chunks, send chunk i to GPU i,
    which concatenates received chunks along the sequence dimension.
    """
    B, local_S, H, D = x.shape
    assert H % world_size == 0, f"H ({H}) must be divisible by world_size ({world_size})"

    # Reshape so each chunk along dim=2 goes to a different GPU
    # [B, S/P, H, D] → [B, S/P, P, H/P, D]
    x = x.view(B, local_S, world_size, H // world_size, D)

    # Permute to [P, B, S/P, H/P, D] so dim-0 is the "destination GPU" axis
    x = x.permute(2, 0, 1, 3, 4).contiguous()

    output = torch.empty_like(x)
    dist.all_to_all_single(output, x)

    # output is [P, B, S/P, H/P, D] — each slot has S/P tokens from one GPU
    # Permute to [B, P, S/P, H/P, D] and merge P * S/P → S
    output = output.permute(1, 0, 2, 3, 4).contiguous()
    output = output.view(B, local_S * world_size, H // world_size, D)
    # Result: [B, S, H/P, D]
    return output


def all_to_all_head_to_seq(x, world_size):
    """Redistribute [B, S, H/P, D] → [B, S/P, H, D] via all-to-all.

    Inverse of all_to_all_seq_to_head.
    """
    B, S, local_H, D = x.shape
    local_S = S // world_size

    # [B, S, H/P, D] → [B, P, S/P, H/P, D]
    x = x.view(B, world_size, local_S, local_H, D)

    # Permute to [P, B, S/P, H/P, D] so dim-0 is the "destination GPU" axis
    x = x.permute(1, 0, 2, 3, 4).contiguous()

    output = torch.empty_like(x)
    dist.all_to_all_single(output, x)

    # output: [P, B, S/P, H/P, D] — reassemble heads
    # Permute to [B, S/P, P, H/P, D] and merge P * H/P → H
    output = output.permute(1, 2, 0, 3, 4).contiguous()
    output = output.view(B, local_S, local_H * world_size, D)
    # Result: [B, S/P, H, D]
    return output


def serial_attention(q, k, v):
    """Compute standard multi-head attention (for verification).

    Args:
        q, k, v: [B, S, H, D]
    Returns:
        [B, S, H, D]
    """
    # Transpose to [B, H, S, D] for scaled_dot_product_attention
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2)  # [B, S, H, D]


def main():
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Configuration
    batch_size = 2
    seq_len = 256
    n_heads = 8
    head_dim = 32
    hidden_dim = n_heads * head_dim  # 256

    assert seq_len % world_size == 0, (
        f"seq_len ({seq_len}) must be divisible by world_size ({world_size})"
    )
    assert n_heads % world_size == 0, (
        f"n_heads ({n_heads}) must be divisible by world_size ({world_size})"
    )
    local_seq_len = seq_len // world_size
    local_heads = n_heads // world_size

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("DeepSpeed-Ulysses Sequence Parallelism")
        print(f"{'=' * 60}")
        print(f"  GPUs:              {world_size}")
        print(f"  Seq length:        {seq_len}")
        print(f"  Heads:             {n_heads}")
        print(f"  Head dim:          {head_dim}")
        print(f"  Hidden dim:        {hidden_dim}")
        print(f"  Local seq (S/P):   {local_seq_len}")
        print(f"  Local heads (H/P): {local_heads}")
        print(f"{'=' * 60}\n")

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Create input and Q/K/V projections
    # ═══════════════════════════════════════════════════════════════
    torch.manual_seed(42)
    Wq = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)
    Wk = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)
    Wv = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)

    # Full input for verification (all GPUs hold the same full tensor)
    full_input = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Each GPU's local chunk of the sequence
    start = rank * local_seq_len
    end = start + local_seq_len
    local_input = full_input[:, start:end, :].contiguous()

    if rank == 0:
        print(f"  STEP 1: Input partitioned along sequence dimension")
        print(f"    Full input:  [{batch_size}, {seq_len}, {hidden_dim}]")
        print(f"    Per GPU:     [{batch_size}, {local_seq_len}, {hidden_dim}]")

    # Project Q, K, V locally (each GPU processes its S/P tokens)
    local_q = Wq(local_input).view(batch_size, local_seq_len, n_heads, head_dim)
    local_k = Wk(local_input).view(batch_size, local_seq_len, n_heads, head_dim)
    local_v = Wv(local_input).view(batch_size, local_seq_len, n_heads, head_dim)

    if rank == 0:
        print(f"    Q/K/V shape: [{batch_size}, {local_seq_len}, {n_heads}, {head_dim}]")

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: all-to-all — sequence-sharded → head-sharded
    # ═══════════════════════════════════════════════════════════════
    # [B, S/P, H, D] → [B, S, H/P, D]
    q_head = all_to_all_seq_to_head(local_q, world_size)
    k_head = all_to_all_seq_to_head(local_k, world_size)
    v_head = all_to_all_seq_to_head(local_v, world_size)

    if rank == 0:
        print(f"\n  STEP 2: all-to-all (sequence-sharded → head-sharded)")
        print(f"    Before: [{batch_size}, {local_seq_len}, {n_heads}, {head_dim}]  "
              f"(S/P tokens, all H heads)")
        print(f"    After:  {list(q_head.shape)}  "
              f"(all S tokens, H/P heads)")
        print(f"    Each GPU now sees the FULL sequence for {local_heads} heads")

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Compute attention — full sequence, subset of heads
    # ═══════════════════════════════════════════════════════════════
    # Each GPU: [B, S, H/P, D] — standard attention on its head subset
    q_attn = q_head.transpose(1, 2)  # [B, H/P, S, D]
    k_attn = k_head.transpose(1, 2)
    v_attn = v_head.transpose(1, 2)
    attn_out = torch.nn.functional.scaled_dot_product_attention(q_attn, k_attn, v_attn)
    attn_out = attn_out.transpose(1, 2)  # [B, S, H/P, D]

    if rank == 0:
        print(f"\n  STEP 3: Attention (full sequence, local heads)")
        print(f"    Attention input:  [{batch_size}, {local_heads}, {seq_len}, {head_dim}]")
        print(f"    Attention output: {list(attn_out.shape)}")
        print(f"    No cross-GPU communication needed during attention!")

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: all-to-all — head-sharded → sequence-sharded
    # ═══════════════════════════════════════════════════════════════
    # [B, S, H/P, D] → [B, S/P, H, D]
    output = all_to_all_head_to_seq(attn_out, world_size)

    if rank == 0:
        print(f"\n  STEP 4: all-to-all (head-sharded → sequence-sharded)")
        print(f"    Before: {list(attn_out.shape)}  "
              f"(all S tokens, H/P heads)")
        print(f"    After:  {list(output.shape)}  "
              f"(S/P tokens, all H heads)")
        print(f"    Back to the original sequence-parallel layout!")

    # ═══════════════════════════════════════════════════════════════
    # STEP 5: Verify against serial (non-parallel) attention
    # ═══════════════════════════════════════════════════════════════
    full_q = Wq(full_input).view(batch_size, seq_len, n_heads, head_dim)
    full_k = Wk(full_input).view(batch_size, seq_len, n_heads, head_dim)
    full_v = Wv(full_input).view(batch_size, seq_len, n_heads, head_dim)
    expected_full = serial_attention(full_q, full_k, full_v)

    # Get expected chunk for this rank
    expected_chunk = expected_full[:, start:end, :, :]  # [B, S/P, H, D]

    match = torch.allclose(output, expected_chunk, atol=1e-5)

    if rank == 0:
        print(f"\n  STEP 5: Verification against serial attention")
        print(f"    Ulysses output matches serial attention: {match}")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    dist.barrier()

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("DeepSpeed-Ulysses Summary")
        print(f"{'=' * 60}")
        print(f"  Communication:  2x all-to-all (not all-gather/reduce-scatter)")
        print(f"  Key idea:       Redistribute between seq-sharded ↔ head-sharded")
        print(f"  Advantage:      Works standalone (no TP needed)")
        print(f"  Requirement:    H must be divisible by P (num GPUs)")
        print(f"")
        print(f"  Data flow:")
        print(f"    [B, S/P, H, D]  ← input: each GPU has S/P tokens")
        print(f"         │")
        print(f"     all-to-all     ← redistribute seq→head")
        print(f"         │")
        print(f"    [B, S, H/P, D]  ← each GPU: full seq, H/P heads")
        print(f"         │")
        print(f"     attention      ← standard MHA on full sequence")
        print(f"         │")
        print(f"    [B, S, H/P, D]  ← attention output")
        print(f"         │")
        print(f"     all-to-all     ← redistribute head→seq")
        print(f"         │")
        print(f"    [B, S/P, H, D]  ← output: back to seq-parallel layout")
        print(f"")
        print(f"  vs Megatron-SP:")
        print(f"    Megatron: all-gather + reduce-scatter, coupled to TP")
        print(f"    Ulysses:  all-to-all + all-to-all, standalone")
        print(f"{'=' * 60}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
