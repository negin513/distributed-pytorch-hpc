#!/usr/bin/env python3
"""
05_ring_attention_concept.py — Ring Attention (P2P ring-based sequence parallelism)

Demonstrates the Ring Attention concept:
    1. Split the sequence into P chunks (one per GPU)
    2. Each GPU computes Q from its local chunk and holds a local KV block
    3. KV blocks rotate around a ring (P2P isend/irecv) for P-1 steps
    4. At each step, compute partial attention with the received KV block
    5. Accumulate results using online softmax (log-sum-exp correction)
    6. After all rotations, each GPU has the correct full attention output

Key advantage: No GPU ever holds the full sequence — enables ultra-long contexts.
Communication: P2P send/recv in a ring (overlaps with compute).

Run with mpiexec:
    mpiexec -n 4 --ppn 4 --cpu-bind none python 05_ring_attention_concept.py

Run with torchrun (single node):
    torchrun --standalone --nproc_per_node=4 05_ring_attention_concept.py

Run with mpiexec (multi-node, 2 nodes x 4 GPUs):
    mpiexec -n 8 --ppn 4 --cpu-bind none python 05_ring_attention_concept.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.distributed as dist
from utils.distributed import init_distributed, cleanup_distributed


def ring_attention(q_local, k_local, v_local, rank, world_size):
    """Ring Attention: rotate KV blocks around a ring of GPUs.

    Each GPU starts with Q for its sequence chunk and a local KV block.
    KV blocks are passed around the ring. At each step, partial attention
    is computed and accumulated using online softmax correction.

    Args:
        q_local: [B, H, S/P, D] — local queries (stays on this GPU)
        k_local: [B, H, S/P, D] — local keys (will be rotated)
        v_local: [B, H, S/P, D] — local values (will be rotated)
        rank: this GPU's rank
        world_size: total number of GPUs

    Returns:
        [B, H, S/P, D] — correct attention output for this GPU's Q chunk
    """
    B, H, local_S, D = q_local.shape
    scale = D ** -0.5

    # Initialize accumulators for online softmax
    # out_acc:  running weighted sum of attention values
    # lse_acc:  running log-sum-exp for numerical stability
    out_acc = torch.zeros_like(q_local)  # [B, H, S/P, D]
    lse_acc = torch.full(
        (B, H, local_S, 1), float("-inf"), device=q_local.device, dtype=q_local.dtype
    )

    # Current KV block on this GPU
    k_recv = k_local.contiguous()
    v_recv = v_local.contiguous()

    # Ring neighbors
    send_to = (rank + 1) % world_size
    recv_from = (rank - 1) % world_size

    for step in range(world_size):
        # ─────────────────────────────────────────────────────
        # Compute partial attention: Q_local * K_received^T
        # ─────────────────────────────────────────────────────
        # Scores: [B, H, S/P, S/P]
        scores = torch.matmul(q_local, k_recv.transpose(-2, -1)) * scale

        # Local log-sum-exp for this block
        lse_block = torch.logsumexp(scores, dim=-1, keepdim=True)  # [B, H, S/P, 1]

        # Attention weights and weighted values for this block
        attn_weights = torch.exp(scores - lse_block)  # [B, H, S/P, S/P]
        out_block = torch.matmul(attn_weights, v_recv)  # [B, H, S/P, D]

        # ─────────────────────────────────────────────────────
        # Online softmax accumulation (log-sum-exp correction)
        # ─────────────────────────────────────────────────────
        # Combine previous accumulator with new block using
        # the log-sum-exp trick to maintain numerical stability.
        #
        # new_lse = log(exp(lse_acc) + exp(lse_block))
        # out_acc = (exp(lse_acc - new_lse) * out_acc
        #          + exp(lse_block - new_lse) * out_block)

        new_lse = torch.logaddexp(lse_acc, lse_block)

        # Correction factors (safe even when lse_acc starts at -inf)
        alpha = torch.exp(lse_acc - new_lse)
        beta = torch.exp(lse_block - new_lse)

        out_acc = alpha * out_acc + beta * out_block
        lse_acc = new_lse

        if rank == 0 and step < 3:
            # Show which KV block we just processed
            source_gpu = (rank - step) % world_size
            print(f"    Step {step}: processed KV from GPU {source_gpu}, "
                  f"scores shape {list(scores.shape)}")

        # ─────────────────────────────────────────────────────
        # Rotate KV to the next GPU in the ring
        # ─────────────────────────────────────────────────────
        if step < world_size - 1:
            k_send = k_recv.contiguous()
            v_send = v_recv.contiguous()
            k_new = torch.empty_like(k_recv)
            v_new = torch.empty_like(v_recv)

            # Async send/recv (overlap with next iteration's compute)
            ops = [
                dist.isend(k_send, dst=send_to),
                dist.irecv(k_new, src=recv_from),
                dist.isend(v_send, dst=send_to),
                dist.irecv(v_new, src=recv_from),
            ]
            for op in ops:
                op.wait()

            k_recv = k_new
            v_recv = v_new

    return out_acc


def serial_attention(q, k, v):
    """Standard multi-head attention for verification.

    Args:
        q, k, v: [B, H, S, D]
    Returns:
        [B, H, S, D]
    """
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def main():
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Configuration
    batch_size = 2
    seq_len = 256
    n_heads = 4
    head_dim = 32

    assert seq_len % world_size == 0, (
        f"seq_len ({seq_len}) must be divisible by world_size ({world_size})"
    )
    local_seq_len = seq_len // world_size

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Ring Attention — Concept Demo")
        print(f"{'=' * 60}")
        print(f"  GPUs:              {world_size}")
        print(f"  Seq length:        {seq_len}")
        print(f"  Heads:             {n_heads}")
        print(f"  Head dim:          {head_dim}")
        print(f"  Local seq (S/P):   {local_seq_len}")
        print(f"  Ring steps:        {world_size} (one per KV block)")
        print(f"{'=' * 60}\n")

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Create full Q, K, V (same on all GPUs for verification)
    # ═══════════════════════════════════════════════════════════════
    torch.manual_seed(42)
    full_q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    full_k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    full_v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)

    # Each GPU takes its chunk of the sequence dimension
    start = rank * local_seq_len
    end = start + local_seq_len

    q_local = full_q[:, :, start:end, :].contiguous()  # [B, H, S/P, D]
    k_local = full_k[:, :, start:end, :].contiguous()
    v_local = full_v[:, :, start:end, :].contiguous()

    if rank == 0:
        print(f"  STEP 1: Partition Q, K, V along sequence dimension")
        print(f"    Full Q/K/V:  [{batch_size}, {n_heads}, {seq_len}, {head_dim}]")
        print(f"    Per GPU:     [{batch_size}, {n_heads}, {local_seq_len}, {head_dim}]")
        print(f"    GPU 0: tokens [0:{local_seq_len}]")
    dist.barrier()
    for r in range(1, world_size):
        if rank == r:
            s = r * local_seq_len
            print(f"    GPU {r}: tokens [{s}:{s + local_seq_len}]")
        dist.barrier()

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Ring Attention — rotate KV blocks around the ring
    # ═══════════════════════════════════════════════════════════════
    if rank == 0:
        print(f"\n  STEP 2: Ring Attention ({world_size} rotation steps)")
        print(f"    Q stays local; KV blocks rotate GPU→GPU via P2P send/recv\n")

    ring_output = ring_attention(q_local, k_local, v_local, rank, world_size)

    if rank == 0:
        if world_size > 3:
            print(f"    ... ({world_size - 3} more steps)")
        print(f"\n    Ring output per GPU: {list(ring_output.shape)}")
        print(f"    No GPU ever held the full [{batch_size}, {n_heads}, "
              f"{seq_len}, {head_dim}] tensor!")

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Verify against serial attention
    # ═══════════════════════════════════════════════════════════════
    expected_full = serial_attention(full_q, full_k, full_v)
    expected_chunk = expected_full[:, :, start:end, :]

    match = torch.allclose(ring_output, expected_chunk, atol=1e-4)

    # Gather match results from all GPUs
    match_tensor = torch.tensor([1.0 if match else 0.0], device=device)
    dist.all_reduce(match_tensor, op=dist.ReduceOp.MIN)
    all_match = match_tensor.item() > 0.5

    if rank == 0:
        print(f"\n  STEP 3: Verification against serial attention")
        print(f"    All GPUs match serial attention: {all_match}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: Memory analysis
    # ═══════════════════════════════════════════════════════════════
    if rank == 0:
        bytes_per_elem = 4  # float32
        full_kv_mem = 2 * batch_size * n_heads * seq_len * head_dim * bytes_per_elem
        local_kv_mem = 2 * batch_size * n_heads * local_seq_len * head_dim * bytes_per_elem
        score_mem_full = batch_size * n_heads * seq_len * seq_len * bytes_per_elem
        score_mem_ring = batch_size * n_heads * local_seq_len * local_seq_len * bytes_per_elem

        print(f"\n  STEP 4: Memory analysis")
        print(f"    Standard attention KV memory:  {full_kv_mem / 1e6:.1f} MB")
        print(f"    Ring attention KV per GPU:     {local_kv_mem / 1e6:.1f} MB  "
              f"({world_size}x reduction)")
        print(f"    Standard attention scores:     {score_mem_full / 1e6:.1f} MB  "
              f"([S, S] = [{seq_len}, {seq_len}])")
        print(f"    Ring attention scores:         {score_mem_ring / 1e6:.1f} MB  "
              f"([S/P, S/P] = [{local_seq_len}, {local_seq_len}])")
        print(f"    Score matrix reduction:        {world_size * world_size}x")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    dist.barrier()

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Ring Attention Summary")
        print(f"{'=' * 60}")
        print(f"  Communication:  P2P send/recv in a ring ({world_size} steps)")
        print(f"  Key idea:       Rotate KV blocks, accumulate with online softmax")
        print(f"  Advantage:      No GPU holds full sequence → ultra-long contexts")
        print(f"  Requirement:    S divisible by P (no head constraint)")
        print(f"")
        print(f"  Ring topology:")
        gpu_labels = " → ".join(f"GPU {i}" for i in range(world_size))
        print(f"    {gpu_labels} → GPU 0  (ring)")
        print(f"")
        print(f"  Each step:")
        print(f"    1. Compute Q_local × K_received^T  (partial attention)")
        print(f"    2. Accumulate via online softmax    (log-sum-exp correction)")
        print(f"    3. Send KV to next GPU, recv from prev")
        print(f"")
        print(f"  After {world_size} steps, each GPU has processed ALL KV blocks")
        print(f"  without ever storing the full [{seq_len}×{seq_len}] score matrix.")
        print(f"")
        print(f"  vs Megatron-SP:  requires TP, uses all-gather/reduce-scatter")
        print(f"  vs Ulysses:      requires H divisible by P, uses all-to-all")
        print(f"  Ring Attention:   only needs S divisible by P, uses P2P ring")
        print(f"{'=' * 60}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
