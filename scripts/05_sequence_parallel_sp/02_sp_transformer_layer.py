#!/usr/bin/env python3
"""
02_sp_transformer_layer.py — Sequence Parallelism with PyTorch APIs

Applies SequenceParallel() to norm layers in a transformer block,
combined with ColwiseParallel/RowwiseParallel for attention/FFN.
Uses PrepareModuleInput for layout transitions between SP and TP regions.

This is the same pattern used in the hybrid example
(scripts/06_hybrid_parallelism/01_fsdp_tp_hybrid.py) but explained
standalone with detailed commentary.

Run with mpiexec:
    mpiexec -n 4 --ppn 4 --cpu-bind none python 02_sp_transformer_layer.py

Run with torchrun (single node):
    torchrun --standalone --nproc_per_node=4 02_sp_transformer_layer.py

Run with mpiexec (multi-node, 2 nodes x 4 GPUs):
    mpiexec -n 8 --ppn 4 --cpu-bind none python 02_sp_transformer_layer.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel,
)

from utils.distributed import init_distributed, cleanup_distributed


# ═══════════════════════════════════════════════════════════════════
# Model: A single transformer block
# ═══════════════════════════════════════════════════════════════════

class SimpleAttention(nn.Module):
    """Multi-head attention with separate Q, K, V projections."""

    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        # Project Q, K, V
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.wo(out)


class FeedForward(nn.Module):
    """SwiGLU-style feed-forward network."""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with attention + FFN."""

    def __init__(self, dim, n_heads):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = SimpleAttention(dim, n_heads)
        self.ffn_norm = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


def main():
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    dim = 256
    n_heads = 8
    seq_len = 128
    batch_size = 2

    assert dim % n_heads == 0
    assert dim % world_size == 0, (
        f"dim ({dim}) must be divisible by world_size ({world_size})"
    )

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Sequence Parallelism on a Transformer Block")
        print(f"{'=' * 60}")
        print(f"  GPUs:       {world_size}")
        print(f"  Dim:        {dim}")
        print(f"  Heads:      {n_heads}")
        print(f"  Seq len:    {seq_len}")
        print(f"  Batch size: {batch_size}")
        print(f"{'=' * 60}\n")

    # ───────────────────────────────────────────────────────────
    # Create a 1D device mesh for TP + SP
    # ───────────────────────────────────────────────────────────
    mesh = init_device_mesh("cuda", (world_size,))

    if rank == 0:
        print("  Device mesh created for TP/SP\n")

    # ───────────────────────────────────────────────────────────
    # Create model and apply TP + SP parallelization
    # ───────────────────────────────────────────────────────────
    block = TransformerBlock(dim, n_heads).to(device)

    if rank == 0:
        param_count = sum(p.numel() for p in block.parameters())
        print(f"  Parameters before parallelization: {param_count:,}\n")

    # The parallelization plan:
    #
    # SequenceParallel() on norm layers:
    #   - Input/output are sharded along S (sequence dim)
    #   - LayerNorm operates locally on each GPU's chunk
    #   - No communication needed for the norm itself
    #
    # PrepareModuleInput for attention:
    #   - Transitions from Shard(1) [SP layout] to Replicate [TP layout]
    #   - This is where the all-gather happens
    #
    # ColwiseParallel on wq, wk, wv:
    #   - Splits the D dimension across GPUs (each GPU: D/N columns)
    #
    # RowwiseParallel on wo:
    #   - Recombines the split D dimension
    #   - output_layouts=Shard(1) keeps output in SP layout

    tp_sp_plan = {
        # Norm layers: operate on sequence-sharded data
        "attention_norm": SequenceParallel(),
        "ffn_norm": SequenceParallel(),

        # Attention: transition SP -> TP
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "attention.wq": ColwiseParallel(),
        "attention.wk": ColwiseParallel(),
        "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),

        # FFN: transition SP -> TP
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        "feed_forward.w3": ColwiseParallel(),
    }

    block = parallelize_module(block, mesh, tp_sp_plan)

    if rank == 0:
        print("  Parallelization plan applied:")
        print("    attention_norm  → SequenceParallel (local LayerNorm)")
        print("    attention       → PrepareModuleInput (SP→TP transition)")
        print("    attention.wq/wk/wv → ColwiseParallel (split heads)")
        print("    attention.wo    → RowwiseParallel (→ SP layout)")
        print("    ffn_norm        → SequenceParallel (local LayerNorm)")
        print("    feed_forward    → PrepareModuleInput (SP→TP transition)")
        print("    feed_forward.w1/w3 → ColwiseParallel")
        print("    feed_forward.w2 → RowwiseParallel (→ SP layout)")
        print()

    # ───────────────────────────────────────────────────────────
    # Run forward pass with sequence-parallel input
    # ───────────────────────────────────────────────────────────
    # Input is sharded along sequence dimension (dim=1)
    local_seq_len = seq_len // world_size
    x = torch.randn(batch_size, local_seq_len, dim, device=device)

    if rank == 0:
        print(f"  Input per GPU: [{batch_size}, {local_seq_len}, {dim}] "
              f"(S/{world_size} tokens)")

    output = block(x)

    if rank == 0:
        print(f"  Output per GPU: {list(output.shape)} "
              f"(still S/{world_size} tokens)")
        print(f"\n  Forward pass successful!")

    # ───────────────────────────────────────────────────────────
    # Backward pass
    # ───────────────────────────────────────────────────────────
    output.sum().backward()

    if rank == 0:
        print(f"  Backward pass successful!")

    # ───────────────────────────────────────────────────────────
    # Summary
    # ───────────────────────────────────────────────────────────
    dist.barrier()

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Data Flow Through the Block")
        print(f"{'=' * 60}")
        print(f"  Input:          [B, S/{world_size}, D]   (SP layout)")
        print(f"  attention_norm: [B, S/{world_size}, D]   (local LayerNorm)")
        print(f"  all-gather:     [B, S, D]        (full sequence)")
        print(f"  wq/wk/wv:      [B, S, D/{world_size}]   (TP split)")
        print(f"  attention:      [B, S, D/{world_size}]   (local compute)")
        print(f"  wo + r-scatter: [B, S/{world_size}, D]   (back to SP)")
        print(f"  ffn_norm:       [B, S/{world_size}, D]   (local LayerNorm)")
        print(f"  all-gather:     [B, S, D]        (full sequence)")
        print(f"  w1/w3:          [B, S, 4D/{world_size}]  (TP split)")
        print(f"  w2 + r-scatter: [B, S/{world_size}, D]   (back to SP)")
        print(f"  Output:         [B, S/{world_size}, D]   (SP layout)")
        print(f"{'=' * 60}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
