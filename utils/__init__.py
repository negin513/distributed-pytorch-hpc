"""
Shared utilities for distributed PyTorch training on Derecho.

Quick start:
    from utils.distributed import init_distributed, cleanup_distributed
    from utils.logging import get_logger, rank_log
"""

from utils.distributed import (
    get_rank_info,
    init_distributed,
    cleanup_distributed,
    is_main_rank,
    print_rank0,
)
from utils.logging import get_logger, rank_log, verify_min_gpu_count
