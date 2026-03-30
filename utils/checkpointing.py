"""
Checkpoint save/load utilities for distributed training.

Handles the common pattern of saving only on rank 0 and loading
on all ranks with proper device mapping.

Usage:
    from utils.checkpointing import save_checkpoint, load_checkpoint

    # Save (only rank 0 writes to disk):
    save_checkpoint(model, optimizer, epoch, "checkpoint.pt", rank)

    # Load (all ranks):
    start_epoch = load_checkpoint(model, optimizer, "checkpoint.pt", device)
"""

import os

import torch
import torch.distributed as dist


def save_checkpoint(model, optimizer, epoch, path, rank=0):
    """
    Save a training checkpoint (model state, optimizer state, epoch).

    Only rank 0 writes to disk. A barrier ensures all ranks wait for
    the save to complete before continuing.

    For DDP-wrapped models, saves ``model.module.state_dict()`` to strip
    the DDP wrapper. For unwrapped models, saves ``model.state_dict()``.

    Args:
        model: The model (possibly DDP-wrapped).
        optimizer: The optimizer.
        epoch: Current epoch number.
        path: File path for the checkpoint.
        rank: Current process rank.
    """
    if rank == 0:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                     exist_ok=True)

        # Handle DDP-wrapped models
        model_state = (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        )

        checkpoint = {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: epoch={epoch}, path={path}")

    # Wait for rank 0 to finish saving
    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(model, optimizer, path, device=None):
    """
    Load a training checkpoint and restore model/optimizer state.

    Args:
        model: The model to load weights into (unwrapped, before DDP wrapping).
        optimizer: The optimizer to restore state into.
        path: File path of the checkpoint.
        device: Device to map tensors to (e.g., ``torch.device("cuda:0")``).

    Returns:
        int: The epoch number stored in the checkpoint, or 0 if file not found.
    """
    if not os.path.exists(path):
        return 0

    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)

    print(f"  Checkpoint loaded: epoch={epoch}, path={path}")
    return epoch
