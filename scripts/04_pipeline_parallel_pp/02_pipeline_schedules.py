#!/usr/bin/env python3
"""
02_pipeline_schedules.py — Pipeline parallelism with PyTorch pipelining API

Demonstrates the torch.distributed.pipelining API (PyTorch 2.4+) to split
a model across GPUs and compare GPipe vs 1F1B scheduling strategies.

GPipe:  All forwards, then all backwards (large bubble, simple)
1F1B:   Interleaved forward/backward (smaller bubble, lower peak memory)

Run with mpiexec:
    mpiexec -n 4 --ppn 4 --cpu-bind none python 02_pipeline_schedules.py

Run with torchrun (single node):
    torchrun --standalone --nproc_per_node=4 02_pipeline_schedules.py

Run with mpiexec (multi-node, 2 nodes x 4 GPUs):
    mpiexec -n 8 --ppn 4 --cpu-bind none python 02_pipeline_schedules.py
"""

import os
import sys
import time

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
# Model: A 4-block MLP that we split across 4 GPUs
# ═══════════════════════════════════════════════════════════════════

class FourBlockMLP(nn.Module):
    """Simple 4-block MLP for demonstrating pipeline splitting."""

    def __init__(self, dim=512):
        super().__init__()
        self.block0 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        self.block1 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        self.block2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        self.block3 = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


def run_schedule(schedule_class, schedule_name, model, device, rank,
                 world_size, num_microbatches, dim, batch_size):
    """Build pipeline stages and run a schedule, returning elapsed time."""

    # Split the model into 4 stages at block boundaries
    # SplitPoint.BEGINNING means "start a new stage BEFORE this module"
    split_spec = {
        "block1": SplitPoint.BEGINNING,
        "block2": SplitPoint.BEGINNING,
        "block3": SplitPoint.BEGINNING,
    }

    # Trace the model to build the pipeline
    example_input = torch.randn(batch_size, dim, device=device)

    pipe = pipeline(
        module=model,
        mb_args=(example_input,),
        split_spec=split_spec,
    )

    # Each rank gets its corresponding stage
    stage = pipe.build_stage(rank, device=device)

    # Create the schedule
    sched = schedule_class(
        stage,
        n_microbatches=num_microbatches,
    )

    # Run forward + backward
    if rank == 0:
        input_data = torch.randn(batch_size, dim, device=device)
    else:
        input_data = None

    torch.cuda.synchronize()
    start = time.time()

    if rank == 0:
        losses = []
        output = sched.step(input_data)
    elif rank == world_size - 1:
        losses = []
        output = sched.step()
    else:
        sched.step()
        output = None

    torch.cuda.synchronize()
    elapsed = time.time() - start

    return elapsed, output


def main():
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if world_size != 4:
        if rank == 0:
            print("This example requires exactly 4 GPUs.")
        cleanup_distributed()
        sys.exit(1)

    dim = 512
    batch_size = 32
    num_microbatches = 4

    if rank == 0:
        print(f"{'=' * 60}")
        print("Pipeline Schedules: GPipe vs 1F1B")
        print(f"{'=' * 60}")
        print(f"  Model:           FourBlockMLP (dim={dim})")
        print(f"  Stages:          {world_size}")
        print(f"  Batch size:      {batch_size}")
        print(f"  Micro-batches:   {num_microbatches}")
        print(f"{'=' * 60}")
        print()

    # ───────────────────────────────────────────────────────────
    # GPipe schedule
    # ───────────────────────────────────────────────────────────
    if rank == 0:
        print("Running GPipe schedule...")
        print("  Schedule: all forwards, then all backwards")

    model_gpipe = FourBlockMLP(dim=dim).to(device)

    gpipe_time, gpipe_output = run_schedule(
        ScheduleGPipe, "GPipe", model_gpipe, device, rank,
        world_size, num_microbatches, dim, batch_size,
    )

    if rank == 0:
        print(f"  GPipe time: {gpipe_time:.4f}s")
        if gpipe_output is not None:
            print(f"  Output shape: {gpipe_output.shape}")
        print()

    dist.barrier()

    # ───────────────────────────────────────────────────────────
    # 1F1B schedule
    # ───────────────────────────────────────────────────────────
    if rank == 0:
        print("Running 1F1B schedule...")
        print("  Schedule: interleaved forward/backward (smaller bubble)")

    model_1f1b = FourBlockMLP(dim=dim).to(device)

    f1b_time, f1b_output = run_schedule(
        Schedule1F1B, "1F1B", model_1f1b, device, rank,
        world_size, num_microbatches, dim, batch_size,
    )

    if rank == 0:
        print(f"  1F1B time: {f1b_time:.4f}s")
        if f1b_output is not None:
            print(f"  Output shape: {f1b_output.shape}")
        print()

    dist.barrier()

    # ───────────────────────────────────────────────────────────
    # Comparison
    # ───────────────────────────────────────────────────────────
    if rank == 0:
        print(f"{'=' * 60}")
        print("COMPARISON")
        print(f"{'=' * 60}")
        print(f"  {'Schedule':<12} {'Time':>10}")
        print(f"  {'─' * 22}")
        print(f"  {'GPipe':<12} {gpipe_time:>10.4f}s")
        print(f"  {'1F1B':<12} {f1b_time:>10.4f}s")
        print()
        print("  Note: Timing differences are more visible with larger")
        print("  models and more micro-batches. The key advantage of 1F1B")
        print("  is lower peak memory (activations freed sooner).")
        print()
        bubble = (world_size - 1) / num_microbatches * 100
        print(f"  Theoretical bubble: {bubble:.0f}% "
              f"({world_size - 1} stages / {num_microbatches} microbatches)")
        print(f"{'=' * 60}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
