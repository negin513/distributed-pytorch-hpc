# Multi-Node PyTorch Distributed Training on NCAR's Derecho
[![Derecho](https://img.shields.io/badge/HPC-Derecho-green)](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/)
[![Last Updated](https://img.shields.io/badge/Last_Updated-March_2026-blue)](https://github.com/NCAR/distributed-pytorch-hpc)

This repository contains example workflows for executing multi-node, multi-GPU machine learning training using PyTorch on NCAR's HPC Supercomputers (i.e. Derecho), along with example PBS scripts for running them.

While this code is written to run directly on [Derecho](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/) GPU nodes, it can be adapted for other GPU HPC machines.

The docs in here are meant to be a quick start guide for users who want to learn more about distributed training paradigms. 

## Quick Start on Derecho

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
