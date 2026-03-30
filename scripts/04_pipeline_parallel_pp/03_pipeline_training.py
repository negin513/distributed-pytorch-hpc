#!/usr/bin/env python3
"""
03_pipeline_training.py — Full training loop with pipeline parallelism

Trains a small transformer model using pipeline parallelism with the
torch.distributed.pipelining API. Loss is computed on the last stage,
and gradients flow back through the pipeline.

Demonstrates:
    - Splitting a transformer into pipeline stages
    - Full forward/backward/optimizer training loop
    - Throughput measurement across stages
    - Configurable number of micro-batches

Run with mpiexec:
    mpiexec -n 4 --ppn 4 --cpu-bind none python 03_pipeline_training.py
    mpiexec -n 4 --ppn 4 --cpu-bind none python 03_pipeline_training.py --num-steps 50

Run with torchrun (single node):
    torchrun --standalone --nproc_per_node=4 03_pipeline_training.py

Run with mpiexec (multi-node, 2 nodes x 4 GPUs):
    mpiexec -n 8 --ppn 4 --cpu-bind none python 03_pipeline_training.py --num-steps 50
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
import torch.distributed as dist
from utils.distributed import init_distributed, cleanup_distributed

from torch.distributed.pipelining import (
    pipeline,
    SplitPoint,
    PipelineStage,
    ScheduleGPipe,
    Schedule1F1B,
)


# ═══════════════════════════════════════════════════════════════════
# Model: A small transformer for pipeline training
# ═══════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """A single transformer encoder block."""

    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm transformer block
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class PipelineTransformer(nn.Module):
    """
    A transformer model designed for 4-stage pipeline splitting.

    The model has 4 named blocks (stage0-stage3), each containing
    multiple transformer layers. The pipeline API splits at block
    boundaries.
    """

    def __init__(self, vocab_size=10000, dim=256, n_heads=8,
                 layers_per_stage=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_encoding = nn.Embedding(1024, dim)

        # 4 blocks of transformer layers (one per pipeline stage)
        self.stage0 = nn.Sequential(
            *[TransformerBlock(dim, n_heads) for _ in range(layers_per_stage)]
        )
        self.stage1 = nn.Sequential(
            *[TransformerBlock(dim, n_heads) for _ in range(layers_per_stage)]
        )
        self.stage2 = nn.Sequential(
            *[TransformerBlock(dim, n_heads) for _ in range(layers_per_stage)]
        )
        self.stage3 = nn.Sequential(
            *[TransformerBlock(dim, n_heads) for _ in range(layers_per_stage)]
        )

        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        # x: [batch, seq_len] token IDs
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoding(positions)

        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.norm(x)
        x = self.output(x)
        return x


def main():
    parser = argparse.ArgumentParser(description="Pipeline Parallel Training")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-microbatches", type=int, default=4)
    parser.add_argument("--schedule", type=str, default="1f1b",
                        choices=["gpipe", "1f1b"])
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers-per-stage", type=int, default=2)
    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if world_size != 4:
        if rank == 0:
            print("This example requires exactly 4 GPUs.")
        cleanup_distributed()
        sys.exit(1)

    vocab_size = 10000
    total_layers = args.layers_per_stage * world_size

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Pipeline Parallel Transformer Training")
        print(f"{'=' * 60}")
        print(f"  Stages:          {world_size}")
        print(f"  Layers/stage:    {args.layers_per_stage}")
        print(f"  Total layers:    {total_layers}")
        print(f"  Model dim:       {args.dim}")
        print(f"  Vocab size:      {vocab_size}")
        print(f"  Batch size:      {args.batch_size}")
        print(f"  Seq length:      {args.seq_len}")
        print(f"  Micro-batches:   {args.num_microbatches}")
        print(f"  Schedule:        {args.schedule.upper()}")
        print(f"  Training steps:  {args.num_steps}")
        print(f"{'=' * 60}\n")

    # ───────────────────────────────────────────────────────────
    # Build model and split into pipeline stages
    # ───────────────────────────────────────────────────────────
    model = PipelineTransformer(
        vocab_size=vocab_size,
        dim=args.dim,
        n_heads=8,
        layers_per_stage=args.layers_per_stage,
    ).to(device)

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {param_count:,}")
        params_per_stage = param_count // world_size
        print(f"  ~Parameters/stage: {params_per_stage:,}\n")

    # Define split points — new stage begins BEFORE each of these modules
    split_spec = {
        "stage1": SplitPoint.BEGINNING,
        "stage2": SplitPoint.BEGINNING,
        "stage3": SplitPoint.BEGINNING,
    }

    # Create example input for tracing
    example_input = torch.randint(0, vocab_size,
                                  (args.batch_size, args.seq_len),
                                  device=device)

    pipe = pipeline(
        module=model,
        mb_args=(example_input,),
        split_spec=split_spec,
    )

    # Build the stage for this rank
    stage = pipe.build_stage(rank, device=device)

    # Choose schedule
    ScheduleClass = Schedule1F1B if args.schedule == "1f1b" else ScheduleGPipe

    # Optimizer (each stage optimizes its own parameters)
    optimizer = torch.optim.AdamW(stage.submod.parameters(), lr=1e-4)

    # Loss function (only used on last stage)
    loss_fn = nn.CrossEntropyLoss()

    # ───────────────────────────────────────────────────────────
    # Training loop
    # ───────────────────────────────────────────────────────────
    if rank == 0:
        print("  Starting training...\n")

    times = []

    for step in range(args.num_steps):
        optimizer.zero_grad()

        # Generate synthetic data
        input_ids = torch.randint(
            0, vocab_size,
            (args.batch_size, args.seq_len),
            device=device,
        )
        target_ids = torch.randint(
            0, vocab_size,
            (args.batch_size, args.seq_len),
            device=device,
        )

        torch.cuda.synchronize()
        start = time.time()

        # Create schedule for this step
        sched = ScheduleClass(
            stage,
            n_microbatches=args.num_microbatches,
            loss_fn=loss_fn,
        )

        # Run the pipeline
        if rank == 0:
            # First stage: provide input
            output = sched.step(input_ids)
        elif rank == world_size - 1:
            # Last stage: provide target for loss
            output = sched.step(target=target_ids)
        else:
            # Middle stages: just pass through
            sched.step()
            output = None

        # Optimizer step
        optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.time() - start

        if step >= 3:  # skip warmup
            times.append(elapsed)

        # Get loss from last stage
        if rank == world_size - 1 and step % 5 == 0:
            # Compute loss for logging
            with torch.no_grad():
                if output is not None:
                    log_loss = loss_fn(
                        output.view(-1, vocab_size), target_ids.view(-1)
                    ).item()
                else:
                    log_loss = float("nan")
            print(f"  Step {step:3d}  loss={log_loss:.4f}  time={elapsed:.3f}s")

    # ───────────────────────────────────────────────────────────
    # Results
    # ───────────────────────────────────────────────────────────
    dist.barrier()

    if rank == 0 and times:
        avg_time = sum(times) / len(times)
        tokens_per_sec = args.batch_size * args.seq_len / avg_time

        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"{'=' * 60}")
        print(f"  Schedule:        {args.schedule.upper()}")
        print(f"  Avg step time:   {avg_time:.3f}s (excluding warmup)")
        print(f"  Throughput:      {tokens_per_sec:,.0f} tokens/sec")
        print(f"  Micro-batches:   {args.num_microbatches}")

        bubble = (world_size - 1) / args.num_microbatches * 100
        print(f"  Bubble fraction: {bubble:.0f}%")
        print(f"{'=' * 60}\n")

    cleanup_distributed()


if __name__ == "__main__":
    main()
