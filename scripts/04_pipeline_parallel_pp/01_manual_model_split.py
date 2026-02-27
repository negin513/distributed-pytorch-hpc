#!/usr/bin/env python3
"""
01_manual_model_split.py — Manual pipeline parallelism with send/recv

Demonstrates the fundamental concept of pipeline parallelism by manually
splitting a 4-layer MLP across 4 GPUs and passing activations between
stages using torch.distributed point-to-point communication.

This is the "from scratch" version — no pipelining API. It shows exactly
what happens under the hood before you use the higher-level API in
02_pipeline_schedules.py.

Run with mpiexec:
    mpiexec -n 4 --ppn 4 --cpu-bind none python 01_manual_model_split.py

Run with torchrun (single node):
    torchrun --standalone --nproc_per_node=4 01_manual_model_split.py

Run multi-node:
    mpiexec -n 8 --ppn 4 --cpu-bind none python 01_manual_model_split.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
import torch.distributed as dist
from utils.distributed import init_distributed, cleanup_distributed


# ═══════════════════════════════════════════════════════════════════
# Model: A simple 4-layer MLP, one layer per GPU
# ═══════════════════════════════════════════════════════════════════

class StageModule(nn.Module):
    """One stage (layer) of the pipeline."""

    def __init__(self, in_features, out_features, is_last=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.is_last = is_last

    def forward(self, x):
        x = self.linear(x)
        if not self.is_last:
            x = torch.relu(x)
        return x


def main():
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if world_size != 4:
        if rank == 0:
            print("This example requires exactly 4 GPUs.")
        cleanup_distributed()
        sys.exit(1)

    if rank == 0:
        print(f"{'=' * 60}")
        print("Manual Pipeline Parallelism (send/recv)")
        print(f"{'=' * 60}")
        print(f"  Stages: {world_size} (one layer per GPU)")
        print()

    # ───────────────────────────────────────────────────────────
    # Each GPU creates only its own stage (layer)
    # ───────────────────────────────────────────────────────────
    # Pipeline: 128 -> 256 -> 256 -> 256 -> 64
    stage_configs = [
        (128, 256, False),  # Stage 0: input layer
        (256, 256, False),  # Stage 1: hidden layer
        (256, 256, False),  # Stage 2: hidden layer
        (256, 64, True),    # Stage 3: output layer
    ]

    in_f, out_f, is_last = stage_configs[rank]
    stage = StageModule(in_f, out_f, is_last=is_last).to(device)

    if rank == 0:
        for i, (inf, outf, _) in enumerate(stage_configs):
            print(f"  GPU {i} (Stage {i}): Linear({inf}, {outf})")
        print()

    # ───────────────────────────────────────────────────────────
    # Forward pass: data flows through the pipeline
    # ───────────────────────────────────────────────────────────
    batch_size = 32
    num_microbatches = 4
    micro_batch_size = batch_size // num_microbatches

    if rank == 0:
        print(f"  Batch size: {batch_size}, Micro-batches: {num_microbatches}")
        print()

    all_outputs = []

    for mb in range(num_microbatches):
        # Stage 0: generate input data and run first layer
        if rank == 0:
            x = torch.randn(micro_batch_size, 128, device=device)
            x = stage(x)
            # Send activations to next stage
            dist.send(x, dst=1)
            if mb == 0:
                print(f"  [Forward] GPU 0: input(8,128) -> stage0 -> send(8,256) to GPU 1")

        # Middle stages: receive, process, send
        elif rank in (1, 2):
            x = torch.empty(micro_batch_size, 256, device=device)
            dist.recv(x, src=rank - 1)
            x = stage(x)
            dist.send(x, dst=rank + 1)
            if mb == 0:
                print(f"  [Forward] GPU {rank}: recv from GPU {rank-1} -> "
                      f"stage{rank} -> send to GPU {rank+1}")

        # Last stage: receive and produce output
        else:  # rank == 3
            x = torch.empty(micro_batch_size, 256, device=device)
            dist.recv(x, src=2)
            x = stage(x)
            all_outputs.append(x)
            if mb == 0:
                print(f"  [Forward] GPU 3: recv from GPU 2 -> "
                      f"stage3 -> output(8,64)")

    # ───────────────────────────────────────────────────────────
    # Only the last stage has the final output
    # ───────────────────────────────────────────────────────────
    if rank == 3:
        full_output = torch.cat(all_outputs, dim=0)
        print(f"\n  Final output shape: {full_output.shape}")
        print(f"  Output on GPU {rank} (last stage)")

    dist.barrier()

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Manual pipeline forward pass complete!")
        print()
        print("Key takeaways:")
        print("  - Each GPU holds only 1/4 of the model parameters")
        print("  - Activations flow GPU 0 -> 1 -> 2 -> 3 via send/recv")
        print("  - This is the foundation of pipeline parallelism")
        print("  - For real training, use the pipelining API (see 02_*)")
        print(f"{'=' * 60}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
