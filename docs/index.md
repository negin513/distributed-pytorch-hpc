# Multi-Node PyTorch Distributed Training on NCAR's Derecho

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/NCAR/distributed-pytorch-hpc/blob/main/LICENSE)
[![Derecho](https://img.shields.io/badge/HPC-Derecho-green)](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/)

This repository contains example workflows for executing multi-node, multi-GPU machine learning training using PyTorch on NCAR's HPC Supercomputers (i.e. Derecho), along with example PBS scripts for running them.

While this code is written to run directly on [Derecho](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/) GPU nodes, it can be adapted for other GPU HPC machines.

## Getting Started

New to distributed training? Start with the [guide](guide/01_single_gpu_baseline.md) — it takes you from a single-GPU training loop to every major distributed strategy.

Already know what you need?

- **[Strategy Decision Guide](guide/strategy_decision_guide.md)** — quick-reference flowchart
- **[Derecho Guide](guide/derecho_guide.md)** — hardware specs and PBS reference
- **[NCCL Tuning](guide/nccl_tuning.md)** — network configuration deep dive
- **[Troubleshooting](guide/troubleshooting.md)** — common errors and solutions

## Quick Start

```bash
git clone https://github.com/NCAR/distributed-pytorch-hpc
cd distributed-pytorch-hpc

module load conda
conda env create -f environment.yml
conda activate pytorch-derecho
```

```bash
# Verify your setup
python -c "import torch; print(f'PyTorch {torch.__version__}, GPUs: {torch.cuda.device_count()}, NCCL: {torch.cuda.nccl.version()}')"
```

```bash
# Run your first distributed job
cd scripts/01_data_parallel_ddp
qsub torchrun_multigpu_ddp.sh -A <your_account>
```
