#!/usr/bin/env python3
"""
03_sp_training.py — Full training with Sequence Parallelism + Tensor Parallelism

Trains a small transformer using SP+TP, with memory comparison showing
the activation savings from sequence parallelism.

Demonstrates:
    - SP+TP applied to a multi-layer transformer
    - Memory usage comparison (with and without SP)
    - Throughput measurement
    - Configurable sequence length to show scaling

Run with mpiexec:
    mpiexec -n 4 --ppn 4 --cpu-bind none python 03_sp_training.py
    mpiexec -n 4 --ppn 4 --cpu-bind none python 03_sp_training.py --seq-len 2048

Run with torchrun (single node):
    torchrun --standalone --nproc_per_node=4 03_sp_training.py

Run with mpiexec (multi-node, 2 nodes x 4 GPUs):
    mpiexec -n 8 --ppn 4 --cpu-bind none python 03_sp_training.py --seq-len 2048
"""

import os
import sys
import time
import argparse

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
# Model: Multi-layer transformer
# ═══════════════════════════════════════════════════════════════════

class Attention(nn.Module):
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
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                               is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = dim * 4
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = Attention(dim, n_heads)
        self.ffn_norm = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class SPTransformer(nn.Module):
    """Transformer with named layers for SP+TP parallelization."""

    def __init__(self, vocab_size, dim, n_heads, n_layers):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, n_heads) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens):
        x = self.tok_embeddings(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)


def apply_sp_tp(model, mesh, rank):
    """Apply SequenceParallel + TensorParallel to the model."""

    # Parallelize embeddings and output
    parallelize_module(
        model, mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),
            ),
        },
    )

    # Parallelize each transformer block
    for layer_id, layer in enumerate(model.layers):
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w3": ColwiseParallel(),
        }
        parallelize_module(layer, mesh, layer_plan)

    if rank == 0:
        print("  SP+TP parallelization applied to all layers")


def main():
    parser = argparse.ArgumentParser(description="SP+TP Training")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10000)
    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    assert args.seq_len % world_size == 0, (
        f"seq_len ({args.seq_len}) must be divisible by world_size ({world_size})"
    )
    assert args.dim % world_size == 0, (
        f"dim ({args.dim}) must be divisible by world_size ({world_size})"
    )

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("SP+TP Transformer Training")
        print(f"{'=' * 60}")
        print(f"  GPUs:         {world_size}")
        print(f"  Layers:       {args.n_layers}")
        print(f"  Dim:          {args.dim}")
        print(f"  Heads:        {args.n_heads}")
        print(f"  Vocab:        {args.vocab_size}")
        print(f"  Seq length:   {args.seq_len}")
        print(f"  Batch size:   {args.batch_size}")
        print(f"  Steps:        {args.num_steps}")
        print(f"{'=' * 60}\n")

    # ───────────────────────────────────────────────────────────
    # Measure baseline memory (before model)
    # ───────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)
    baseline_mem = torch.cuda.memory_allocated(device)

    # ───────────────────────────────────────────────────────────
    # Create model with SP+TP
    # ───────────────────────────────────────────────────────────
    mesh = init_device_mesh("cuda", (world_size,))

    model = SPTransformer(
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {param_count:,}")

    apply_sp_tp(model, mesh, rank)

    model_mem = torch.cuda.memory_allocated(device) - baseline_mem

    if rank == 0:
        print(f"  Model memory per GPU: {model_mem / 1e6:.1f} MB")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # ───────────────────────────────────────────────────────────
    # Training loop
    # ───────────────────────────────────────────────────────────
    if rank == 0:
        print(f"\n  Starting training...\n")

    times = []
    torch.cuda.reset_peak_memory_stats(device)

    for step in range(args.num_steps):
        # Each TP group gets the same input tokens (seeded by step)
        torch.manual_seed(step)
        input_ids = torch.randint(
            0, args.vocab_size,
            (args.batch_size, args.seq_len),
            device=device,
        )
        target_ids = torch.randint(
            0, args.vocab_size,
            (args.batch_size, args.seq_len),
            device=device,
        )

        torch.cuda.synchronize()
        start = time.time()

        optimizer.zero_grad()

        # Forward — embeddings produce SP layout automatically
        logits = model(input_ids)  # [B, S, vocab] — Replicated via output plan

        loss = loss_fn(logits.view(-1, args.vocab_size), target_ids.view(-1))
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.time() - start

        if step >= 3:
            times.append(elapsed)

        if rank == 0 and step % 5 == 0:
            print(f"  Step {step:3d}  loss={loss.item():.4f}  time={elapsed:.3f}s")

    # ───────────────────────────────────────────────────────────
    # Results
    # ───────────────────────────────────────────────────────────
    peak_mem = torch.cuda.max_memory_allocated(device)

    dist.barrier()

    if rank == 0 and times:
        avg_time = sum(times) / len(times)
        tokens_per_sec = args.batch_size * args.seq_len / avg_time

        # Estimate memory without SP (full activations on each GPU)
        local_seq = args.seq_len // world_size
        act_per_layer_sp = args.batch_size * local_seq * args.dim * 4  # bytes
        act_per_layer_no_sp = args.batch_size * args.seq_len * args.dim * 4
        act_savings = (act_per_layer_no_sp - act_per_layer_sp) * args.n_layers * 2

        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"{'=' * 60}")
        print(f"  Avg step time:    {avg_time:.3f}s (excluding warmup)")
        print(f"  Throughput:       {tokens_per_sec:,.0f} tokens/sec")
        print(f"  Peak GPU memory:  {peak_mem / 1e6:.1f} MB")
        print()
        print(f"  MEMORY ANALYSIS (activations per norm/dropout layer):")
        print(f"    Without SP:  [{args.batch_size}, {args.seq_len}, {args.dim}] "
              f"= {act_per_layer_no_sp / 1e6:.1f} MB")
        print(f"    With SP:     [{args.batch_size}, {local_seq}, {args.dim}] "
              f"= {act_per_layer_sp / 1e6:.1f} MB")
        print(f"    Reduction:   {world_size}x per layer")
        print(f"    Est. total savings: ~{act_savings / 1e6:.1f} MB "
              f"({args.n_layers} layers x 2 norms)")
        print(f"{'=' * 60}\n")

    cleanup_distributed()


if __name__ == "__main__":
    main()
