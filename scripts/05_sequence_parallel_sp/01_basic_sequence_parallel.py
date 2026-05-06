#!/usr/bin/env python3
"""
01_basic_sequence_parallel.py — Sequence parallelism fundamentals

Demonstrates the core concepts of sequence parallelism:
    1. Splitting a [B, S, D] tensor along the sequence dimension S
    2. LayerNorm works correctly on local chunks (no communication)
    3. All-gather reconstructs the full sequence before attention
    4. Reduce-scatter distributes results back after attention

This uses manual collectives to show exactly what happens. The next
example (02_*) uses PyTorch's SequenceParallel() API.

Run with mpiexec:
    mpiexec -n 4 --ppn 4 --cpu-bind none python 01_basic_sequence_parallel.py

Run with torchrun (single node):
    torchrun --standalone --nproc_per_node=4 01_basic_sequence_parallel.py

Run with mpiexec (multi-node, 2 nodes x 4 GPUs):
    mpiexec -n 8 --ppn 4 --cpu-bind none python 01_basic_sequence_parallel.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
import torch.distributed as dist
from utils.distributed import init_distributed, cleanup_distributed


def main():
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    batch_size = 2
    seq_len = 256
    hidden_dim = 128

    assert seq_len % world_size == 0, (
        f"seq_len ({seq_len}) must be divisible by world_size ({world_size})"
    )
    local_seq_len = seq_len // world_size

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Sequence Parallelism — Basic Concepts")
        print(f"{'=' * 60}")
        print(f"  GPUs:            {world_size}")
        print(f"  Full shape:      [{batch_size}, {seq_len}, {hidden_dim}]")
        print(f"  Local shape:     [{batch_size}, {local_seq_len}, {hidden_dim}]")
        print(f"{'=' * 60}\n")

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Create the full tensor on all GPUs (for verification)
    # ═══════════════════════════════════════════════════════════════
    torch.manual_seed(42)
    full_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Each GPU takes its chunk of the sequence dimension
    start = rank * local_seq_len
    end = start + local_seq_len
    local_chunk = full_tensor[:, start:end, :].contiguous()

    if rank == 0:
        print(f"  STEP 1: Split sequence across GPUs")
        print(f"    GPU 0: tokens [{start}:{end}]")
    dist.barrier()
    for r in range(world_size):
        if rank == r and rank > 0:
            s = r * local_seq_len
            e = s + local_seq_len
            print(f"    GPU {r}: tokens [{s}:{e}]")
        dist.barrier()

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: LayerNorm works locally — no communication needed
    # ═══════════════════════════════════════════════════════════════
    layer_norm = nn.LayerNorm(hidden_dim).to(device)

    # Apply LayerNorm to the local chunk
    local_normed = layer_norm(local_chunk)

    # Apply LayerNorm to the full tensor for comparison
    full_normed = layer_norm(full_tensor)
    expected_chunk = full_normed[:, start:end, :]

    # Verify they match
    match = torch.allclose(local_normed, expected_chunk, atol=1e-6)

    if rank == 0:
        print(f"\n  STEP 2: LayerNorm on local chunks")
        print(f"    LayerNorm normalizes over D (last dim) per token")
        print(f"    Each GPU has complete D-vectors, just fewer tokens")
        print(f"    Local result matches full result: {match}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: All-gather to reconstruct full sequence
    # ═══════════════════════════════════════════════════════════════
    # Before attention, we need the full sequence on each GPU
    # (attention computes Q*K^T across all tokens)
    gathered_chunks = [
        torch.empty_like(local_normed) for _ in range(world_size)
    ]
    dist.all_gather(gathered_chunks, local_normed)
    gathered_full = torch.cat(gathered_chunks, dim=1)  # [B, S, D]

    gather_match = torch.allclose(gathered_full, full_normed, atol=1e-6)

    if rank == 0:
        print(f"\n  STEP 3: All-gather before attention")
        print(f"    Each GPU: [{batch_size}, {local_seq_len}, {hidden_dim}]")
        print(f"    After all-gather: [{batch_size}, {seq_len}, {hidden_dim}]")
        print(f"    Matches full LayerNorm output: {gather_match}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: Simulate attention (needs full sequence)
    # ═══════════════════════════════════════════════════════════════
    # Simple linear projection as a stand-in for attention
    proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)
    attn_output = proj(gathered_full)  # [B, S, D]

    if rank == 0:
        print(f"\n  STEP 4: Attention (on full sequence)")
        print(f"    Input:  [{batch_size}, {seq_len}, {hidden_dim}]")
        print(f"    Output: {list(attn_output.shape)}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 5: Reduce-scatter to split output back along sequence
    # ═══════════════════════════════════════════════════════════════
    # After attention, split the output back so each GPU holds S/N tokens
    # reduce_scatter sums partial results AND scatters in one operation
    scatter_input = list(attn_output.chunk(world_size, dim=1))
    scatter_output = torch.empty(
        batch_size, local_seq_len, hidden_dim, device=device
    )
    dist.reduce_scatter(scatter_output, scatter_input)

    if rank == 0:
        print(f"\n  STEP 5: Reduce-scatter after attention")
        print(f"    Full output: [{batch_size}, {seq_len}, {hidden_dim}]")
        print(f"    After reduce-scatter: [{batch_size}, {local_seq_len}, {hidden_dim}]")
        print(f"    Back to sequence-parallel layout!")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    dist.barrier()

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Summary: Sequence Parallel Data Flow")
        print(f"{'=' * 60}")
        print(f"  [B, S/N, D]  ← LayerNorm (local, no communication)")
        print(f"       │")
        print(f"   all-gather  ← reconstruct full sequence")
        print(f"       │")
        print(f"  [B, S, D]   ← Attention (needs full sequence)")
        print(f"       │")
        print(f"  reduce-scatter  ← split back along S")
        print(f"       │")
        print(f"  [B, S/N, D]  ← FFN Norm, Dropout (local)")
        print(f"")
        print(f"  Memory saved: {world_size}x on LayerNorm/Dropout activations")
        print(f"{'=' * 60}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
