"""
PyTorch profiler utilities for distributed training.

Wraps ``torch.profiler.profile`` with sensible defaults for GPU workloads
on Derecho (CUDA + CPU activities, Chrome trace export).

Usage:
    from utils.profiling import training_profiler, print_profiler_summary

    with training_profiler(output_dir="./profiler_output") as prof:
        for step in range(num_steps):
            train_step()
            prof.step()

    print_profiler_summary(prof, rank)
"""

import os
from contextlib import contextmanager

import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler


@contextmanager
def training_profiler(output_dir="./profiler_output", wait=1, warmup=1,
                      active=3, repeat=1, rank=0):
    """
    Context manager wrapping ``torch.profiler.profile`` with CUDA+CPU activities.

    Generates Chrome trace files in ``output_dir`` (only on rank 0 by default).

    Args:
        output_dir: Directory for trace output files.
        wait: Number of steps to skip before profiling.
        warmup: Number of warmup steps (profiler active but results discarded).
        active: Number of steps to actively profile.
        repeat: Number of profiling cycles.
        rank: Current process rank (traces saved only on rank 0).

    Yields:
        torch.profiler.profile instance (call ``.step()`` after each training step).
    """
    os.makedirs(output_dir, exist_ok=True)

    prof_schedule = schedule(
        wait=wait, warmup=warmup, active=active, repeat=repeat
    )

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    trace_handler = None
    if rank == 0:
        trace_handler = tensorboard_trace_handler(output_dir)

    with profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        yield prof


def print_profiler_summary(prof, rank=0, top_n=10):
    """
    Print a summary table of the most time-consuming operations.

    Args:
        prof: A completed ``torch.profiler.profile`` instance.
        rank: Current process rank (prints only on rank 0).
        top_n: Number of top operations to show.
    """
    if rank != 0:
        return

    print(f"\n{'=' * 70}")
    print("Profiler Summary (top operations by CUDA time)")
    print(f"{'=' * 70}")
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=top_n
    ))
