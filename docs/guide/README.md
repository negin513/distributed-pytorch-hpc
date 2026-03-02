# From Basics to Distributed: A Practical Guide to Multi-GPU Training

This guide takes you from a single-GPU PyTorch training loop to every major
distributed training strategy, with a focus on running on NCAR's Derecho
supercomputer. Each chapter builds on the previous one, so reading in order
is recommended for newcomers.

## Prerequisites

- Comfortable with PyTorch basics: `nn.Module`, `DataLoader`, training loops
- Access to a multi-GPU system (Derecho or similar HPC cluster)
- Conda environment set up per the repo's `environment.yml`

## Chapters

| # | Chapter | What You'll Learn |
|---|---------|-------------------|
| 01 | [Single-GPU Baseline](01_single_gpu_baseline.md) | The starting point: a complete training script and where the limits are |
| 02 | [Why Distributed?](02_why_distributed.md) | The three walls you'll hit and the four things you can split |
| 03 | [Communication Primitives](03_communication_primitives.md) | Ranks, process groups, and the five collective operations |
| 04 | [Data Parallel (DDP)](04_data_parallel_ddp.md) | Your first distributed strategy — replicate model, split data |
| 05 | [Fully Sharded (FSDP)](05_fully_sharded_fsdp.md) | Shard parameters, gradients, and optimizer state across GPUs |
| 06 | [Tensor Parallel](06_tensor_parallel.md) | Split individual weight matrices across GPUs |
| 07 | [Pipeline Parallel](07_pipeline_parallel.md) | Split model layers across GPUs and pipeline micro-batches |
| 08 | [Sequence Parallel](08_sequence_parallel.md) | Handle sequences too long for a single GPU |
| 09 | [Hybrid Parallelism](09_hybrid_parallelism.md) | Combine TP + FSDP for large language models |
| 10 | [Domain Parallel](10_domain_parallel.md) | Split spatial data for climate, weather, and physics models |
| 11 | [Choosing a Strategy](11_choosing_a_strategy.md) | Decision flowchart and concrete recommendations |
| 12 | [HPC Operations](12_hpc_operations.md) | PBS jobs, NCCL tuning, and debugging on Derecho |

## How to Use This Guide

- **New to distributed training?** Start at Chapter 01 and read sequentially.
- **Know the basics, need a specific strategy?** Jump to Chapters 04-10.
- **Setting up a job on Derecho?** Go straight to Chapter 12.
- **Not sure which strategy to use?** Chapter 11 has a decision flowchart.

Each chapter links to runnable scripts in the `scripts/` directory. The guide
explains the *why* and *how*; the scripts are the working *what*.

## Related Resources

- [Strategy Decision Guide](../strategy_decision_guide.md) — quick-reference flowchart
- [Derecho Guide](../derecho_guide.md) — hardware specs and PBS reference
- [NCCL Tuning](../nccl_tuning.md) — network configuration deep dive
- [Troubleshooting](../troubleshooting.md) — common errors and solutions
