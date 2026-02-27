"""
Rank-aware logging utilities for distributed training.

Replaces scripts/06_hybrid_parallelism/log_utils.py and inline
rank-conditional prints throughout the codebase.

Usage:
    from utils.logging import get_logger, rank_log

    logger = get_logger()
    rank_log(rank, logger, "Training started")
"""

import logging

import torch


logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def get_logger(name=None):
    """
    Return a logger instance.

    When used in distributed training, combine with ``rank_log``
    to ensure only rank 0 emits messages.

    Args:
        name: Logger name. Defaults to the calling module's ``__name__``.

    Returns:
        logging.Logger
    """
    return logging.getLogger(name or __name__)


def rank_log(rank, logger, msg):
    """
    Log a message only from rank 0.

    Args:
        rank: Current process rank.
        logger: A ``logging.Logger`` instance.
        msg: Message string to log.
    """
    if rank == 0:
        logger.info(f" {msg}")


def verify_min_gpu_count(min_gpus=2):
    """
    Check that the system has at least ``min_gpus`` CUDA devices.

    Args:
        min_gpus: Minimum number of GPUs required.

    Returns:
        bool: True if enough GPUs are available.
    """
    return torch.cuda.is_available() and torch.cuda.device_count() >= min_gpus
