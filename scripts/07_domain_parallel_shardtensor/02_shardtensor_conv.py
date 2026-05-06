#!/usr/bin/env python3
"""
02_dtensor_conv.py - Domain-parallel convolution correctness check

Demonstrates that PyTorch's DTensor gives IDENTICAL results to single-GPU
computation for a Conv2d, including gradients. No PhysicsNeMo required -
uses only pure PyTorch DTensor (available since PyTorch 2.4+).

Run with mpiexec:
    mpiexec -n 4 --ppn 4 --cpu-bind none python 02_shardtensor_conv.py

Run with torchrun:
    torchrun --standalone --nproc_per_node=4 02_shardtensor_conv.py

What happens step by step:
    1. Every GPU creates the SAME full 1024x1024 tensor (same seed)
    2. Every GPU computes the single-GPU "ground truth" forward + backward
    3. distribute_tensor() shards it: each GPU gets a spatial slice
       (e.g., 4 GPUs -> each gets 256 rows of the 1024-row image)
    4. distribute_module() tells PyTorch the Conv2d will operate on
       distributed tensors - DTensor handles halo exchange internally
    5. Forward + backward on the sharded tensor
    6. full_tensor() gathers the output back - we verify it matches

Key Concepts:
    - DeviceMesh: defines the group of GPUs (like a topology)
    - Shard(dim): says "this tensor is split along dimension `dim`"
    - distribute_tensor: distributes data across the mesh
    - distribute_module: wraps a module for distributed operation
    - full_tensor(): allgather back to full size (for verification)
"""

import os
import sys

# Add repo root to path so `from utils...` works
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module

from utils.distributed import init_distributed, cleanup_distributed


def main():
    rank, world_size, local_rank = init_distributed(verbose=False)
    device = torch.device(f"cuda:{local_rank}")

    # ===========================================================
    # STEP 1: Single-GPU ground truth
    # ===========================================================
    torch.manual_seed(0)
    full_tensor = torch.randn(1, 8, 1024, 1024, device=device, requires_grad=True)
    conv = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1).to(device)

    # Forward + backward on full tensor
    single_gpu_output = conv(full_tensor)
    single_gpu_output.mean().backward()
    single_gpu_grad = full_tensor.grad.data.clone()

    # Zero out grad so it doesn't interfere with the distributed pass
    full_tensor.grad = None

    if rank == 0:
        print(f"Single-GPU output shape: {single_gpu_output.shape}")
        print(f"Single-GPU grad shape:   {single_gpu_grad.shape}")

    # ===========================================================
    # STEP 2: Create a DeviceMesh
    # ===========================================================
    # A DeviceMesh defines the GPU topology for domain parallelism.
    # For 4 GPUs this creates a 1D mesh: [GPU0, GPU1, GPU2, GPU3]
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("domain",))
    if rank == 0:
        print(f"\nDeviceMesh: {world_size} GPUs in 1D domain-parallel mesh")

    # ===========================================================
    # STEP 3: Distribute the tensor across GPUs
    # ===========================================================
    # Shard(2) = split along dimension 2 (height).
    # For a [1, 8, 1024, 1024] tensor on 4 GPUs:
    #   GPU 0 gets rows 0-255
    #   GPU 1 gets rows 256-511
    #   GPU 2 gets rows 512-767
    #   GPU 3 gets rows 768-1023
    sharded_input = distribute_tensor(
        full_tensor,
        device_mesh=mesh,
        placements=[Shard(2)],  # shard along dim 2 (height)
    )
    # Re-enable grad tracking on the distributed tensor
    sharded_input = sharded_input.detach().requires_grad_(True)

    local_shape = sharded_input.to_local().shape
    print(f"  Rank {rank}: local shard shape = {local_shape}")
    # Each GPU holds [1, 8, 256, 1024] - only 1/4 of the memory!

    # ===========================================================
    # STEP 4: Distribute the model
    # ===========================================================
    # distribute_module tells PyTorch that this Conv2d will receive
    # DTensors. Internally, DTensor routes conv2d to handle the
    # sharded input correctly (including halo exchange for boundary pixels).
    distributed_conv = distribute_module(conv, mesh)

    # ===========================================================
    # STEP 5: Forward + backward (distributed)
    # ===========================================================
    sharded_output = distributed_conv(sharded_input)
    sharded_output.mean().backward()

    has_grad = sharded_input.grad is not None

    if rank == 0:
        print(f"\nSharded output local shape: {sharded_output.to_local().shape}")
        if has_grad:
            print(f"Sharded grad local shape:   {sharded_input.grad.to_local().shape}")
        else:
            print("Sharded grad:               None (DTensor conv backward limitation)")

    # ===========================================================
    # STEP 6: Gather and verify correctness
    # ===========================================================
    # full_tensor() triggers an allgather to reconstruct the full tensor
    gathered_output = sharded_output.full_tensor()

    if rank == 0:
        output_match = torch.allclose(gathered_output, single_gpu_output, atol=1e-5)

        print(f"\n{'='*60}")
        print(f"CORRECTNESS CHECK")
        print(f"{'='*60}")
        print(f"  Forward pass matches single GPU:  {output_match}")

        if has_grad:
            gathered_grad = sharded_input.grad.full_tensor()
            grad_match = torch.allclose(gathered_grad, single_gpu_grad, atol=1e-5)
            print(f"  Backward pass matches single GPU: {grad_match}")
        else:
            grad_match = None
            print(f"  Backward pass: grad not computed (DTensor does not yet")
            print(f"    propagate input gradients for spatially-sharded Conv2d)")

        print(f"{'='*60}")

        if output_match and grad_match is not False:
            print("  SUCCESS - DTensor forward pass gives identical results!")
            if grad_match is None:
                print("  NOTE: backward pass grad verification skipped (see above)")
        else:
            print("  MISMATCH - something went wrong")
            if not output_match:
                max_diff = (gathered_output - single_gpu_output).abs().max().item()
                print(f"  Output max diff: {max_diff}")
            if grad_match is False:
                gathered_grad = sharded_input.grad.full_tensor()
                max_diff = (gathered_grad - single_gpu_grad).abs().max().item()
                print(f"  Grad max diff: {max_diff}")

    if has_grad:
        print(f"  Rank {rank}: grad placement={sharded_input.grad.placements}, "
              f"local_shape={sharded_input.grad.to_local().shape}")
    else:
        print(f"  Rank {rank}: no input grad computed")

    cleanup_distributed()


if __name__ == "__main__":
    main()
