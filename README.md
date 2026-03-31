# Multi-Node PyTorch Distributed Training on NCAR's Derecho

[![Last Updated](https://img.shields.io/badge/Last_Updated-March_2026-blue)](https://github.com/NCAR/distributed-pytorch-hpc)
[![Docs](https://img.shields.io/badge/Docs-online-blue)](https://negin513.github.io/distributed-pytorch-hpc/)

## Overview

This repostory contains a collecion of example workflows for executing multi-node, multi-GPU machine learning training using PyTorch on NSF NCAR's HPC Supercomputers (i.e. Derecho), along with example PBS scripts for running them.

While this code is written to run directly on [Derecho](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/) GPU nodes, it can be adapted for other GPU HPC machines. Each [Derecho](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/) node has 4 NVIDIA A100 GPUs connected by NVLink, and nodes are connected via the HPE Slingshot interconnect.

The goal is to provide a starting point for researchers who want to scale their PyTorch training workflow to multiple GPUs and nodes on NCAR's HPC systems using different distributed training paradigms. 

## Contents



--------------------------------------------------------------
## What is Distributed Training?

Distributed training allows you to train AI models across multiple GPUs, enabling you to scale up to larger models and datasets than a single GPU can handle. PyTorch provides several built-in strategies for distributed training, each with its own tradeoffs in terms of memory usage, communication overhead, and ease of implementation. 

In this repository, we cover the following strategies:

### What is DDP (Distributed Data Parallel)?

[Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) is PyTorch's most widely used distributed training strategy. Each GPU holds a **complete copy** of the model and processes a different slice of the training data. After each backward pass, gradients are synchronized across all GPUs using an all-reduce operation, ensuring every replica stays in sync. DDP scales your effective batch size linearly.
**Start here** if your model fits on a single GPU. It's the simplest strategy and often the fastest.
See [`scripts/01_data_parallel_ddp/`](scripts/01_data_parallel_ddp/) for examples.

### What is FSDP (Fully Sharded Data Parallelism)?

[Fully Sharded Data Parallel (FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) solves the main limitation of DDP: memory. Instead of keeping a full copy of the model on every GPU, FSDP **shards** the model parameters, gradients, and optimizer states across GPUs. Each GPU only stores 1/N of the total state (where N = number of GPUs). When a layer needs its full parameters (during forward/backward), FSDP temporarily all-gathers them from other GPUs, computes, then discards the non-local shards. This trades communication for memory, enabling models that are 4-8x larger than what DDP can handle.

See [`scripts/02_fully_sharded_fsdp/`](scripts/02_fully_sharded_fsdp/) for examples. For a deeper comparison of DDP vs FSDP, see [this article](https://openmmlab.medium.com/its-2023-is-pytorch-s-fsdp-the-best-choice-for-training-large-models-fe8d2848832f).


### More Advanced Strategies: TP, PP, SP, Hybrid, and Domain Parallelism

### What is Tensor Parallelism (TP)?

[Tensor Parallelism](https://pytorch.org/tutorials/intermediate/TP_tutorial.html) splits individual **weight matrices** across GPUs. Instead of each GPU having a full copy of a large linear layer, the weight matrix is divided column-wise or row-wise so each GPU computes a portion of the output. TP is most effective for models with very large individual layers (e.g., the attention and FFN layers in LLMs). It requires high-bandwidth GPU interconnect since activations are communicated at every layer.

See [`scripts/03_tensor_parallel_tp/`](scripts/03_tensor_parallel_tp/) for examples.

### What is Pipeline Parallelism (PP)?

Pipeline Parallelism splits a model **by layers** across GPUs, forming a pipeline of stages. GPU 0 runs layers 0-9, GPU 1 runs layers 10-19, and so on. The training batch is split into microbatches that flow through the pipeline, allowing multiple GPUs to be active simultaneously. PP uses point-to-point send/recv (not all-reduce), so it has low communication overhead. The tradeoff is the pipeline bubble — stages sit idle while the pipeline fills and drains. More microbatches reduce the bubble. Best for very deep models (100+ layers).

See [`scripts/04_pipeline_parallel_pp/`](scripts/04_pipeline_parallel_pp/) for examples.

### What is Sequence Parallelism (SP)?

Sequence Parallelism splits the **sequence dimension** of activations across GPUs. In transformer models, the attention mechanism produces activations of shape `(batch, sequence_length, hidden_dim)`. When sequence lengths are very long, these activations dominate memory usage. SP distributes this cost across GPUs. SP is typically combined with TP — TP handles the attention/FFN computation while SP handles the LayerNorm and dropout operations on split sequences. This combination gives memory savings on both parameters and activations.

See [`scripts/05_sequence_parallel_sp/`](scripts/05_sequence_parallel_sp/) for examples.

### What is Hybrid Parallelism (TP + FSDP)?

For the largest models, a single strategy isn't enough. Hybrid parallelism combines TP within a node (where GPU bandwidth is highest) with FSDP across nodes (where communication cost is higher but less frequent). This is the standard approach for training foundation models at scale. PyTorch's `DeviceMesh` API makes it straightforward to define the 2D mesh of (TP, FSDP) dimensions.

See [`scripts/06_hybrid_parallelism/`](scripts/06_hybrid_parallelism/) for examples.

### What is Domain Parallelism?
In Scientific AI, we often have large 3D spatial domains that we want to model. Domain Parallelism splits the **spatial domain** across GPUs, so each GPU is responsible for a different chunk of the 3D grid. This is common in climate and weather models. Communication happens at the boundaries of the domain chunks to exchange halo data. This approach can be combined with DDP or FSDP for the model parallelism within each domain chunk.

See [`scripts/07_domain_parallelism/`](scripts/07_domain_parallelism/) for examples.

---------------------------------------

## Quick Start

### 1. Setup Environment

```bash
git clone https://github.com/NCAR/distributed-pytorch-hpc
cd distributed-pytorch-hpc

module load conda

# create environment from custom torch wheel built for Derecho (with NCCL tuned for Slingshot)
conda env create -f environment.yml
conda activate pytorch-derecho
```

### 2. Run DDP Example
``` bash
# submit examples to train on Derecho using PBS scripts in `scripts/`
cd scripts/01_data_parallel_ddp
qsub run_ddp.sh -A <your_account>  # submit DDP job to PBS
```

## Launching Distributed Jobs

In distributed machine learning, a **launcher** is the tool or command that starts your training processes across one or more compute nodes.
A launcher takes care of the following:

- Starting the correct number of processes across nodes and GPUs
- Setting environment variables needed for distributed communication
  (e.g., `RANK`, `WORLD_SIZE`, `MASTER_ADDR`)
- Coordinating process startup and synchronization

Your training code then uses these environment variables to initialize the process group and set up distributed training.

### Single Node Launching with `torchrun`
For single-node multi-GPU training, PyTorch's built-in `torchrun` is the simplest option. It automatically sets up the environment variables and spawns one process per GPU. For example, to train on 4 GPUs on a single node:
```bash
torchrun --nproc_per_node=4 train.py
```

On Derecho, we recommend launching distributed jobs using **`mpiexec`** or
**`torchrun`**. Using `mpiexec` integrates natively with PBS and the
Slingshot interconnect. `torchrun` is PyTorch's built-in launcher and is
convenient for single-node testing. All examples in this repo work with
any of these approaches — `utils/distributed.py` auto-detects the launcher
and sets up ranks accordingly.

### Option 1: `torchrun` (single node only)

`torchrun` is PyTorch's built-in launcher. It spawns one process per GPU,
sets `LOCAL_RANK` / `RANK` / `WORLD_SIZE` environment variables, and handles
`MASTER_ADDR` / `MASTER_PORT` automatically.

```bash
# Single node, 4 GPUs — simplest way to get started
torchrun --standalone --nproc_per_node=4 train.py
```

**Pros:** No MPI dependency, built into PyTorch, simple single-node usage.
**Cons:** Cannot launch across multiple nodes by itself on Derecho — needs
`mpiexec` to distribute across nodes.

### Option 2: `mpiexec` + `torchrun` (multi-node)

On Derecho, `mpiexec` handles multi-node process placement over Slingshot,
and `torchrun` manages per-node GPU processes. This launches one `torchrun`
per node with `--ppn 1` (1 MPI rank per node), and `torchrun` forks 4 GPU
workers on that node.

```bash
# 2 nodes x 4 GPUs = 8 GPUs total
# mpiexec places 1 rank per node, torchrun spawns 4 GPU workers per node
NNODES=$(< $PBS_NODEFILE wc -l)
HEAD_NODE_IP=$(ssh $(head -1 $PBS_NODEFILE) hostname -i | awk '{print $1}')

mpiexec -n $NNODES --ppn 1 --cpu-bind none \
    torchrun \
        --nnodes=$NNODES \
        --nproc-per-node=4 \
        --rdzv-backend=c10d \
        --rdzv-endpoint=$HEAD_NODE_IP \
        train.py
```

**Pros:** Robust multi-node support, torchrun handles rendezvous.
**Cons:** More verbose, requires node discovery for `--rdzv-endpoint`.

### Option 3: `mpiexec` alone (recommended on Derecho)

`mpiexec` launches one process per GPU directly — no `torchrun` involved.
The script detects its rank from MPI environment variables (`OMPI_*` or
`PMI_RANK`) via `utils/distributed.py`. This is the approach used by
all PBS scripts in this repo.

```bash
# Single node, 4 GPUs
mpiexec -n 4 --ppn 4 --cpu-bind none python train.py

# 2 nodes x 4 GPUs = 8 GPUs total
mpiexec -n 8 --ppn 4 --cpu-bind none python train.py

# 4 nodes x 4 GPUs = 16 GPUs total
mpiexec -n 16 --ppn 4 --cpu-bind none python train.py
```

**Pros:** Single command for both single-node and multi-node, integrates
natively with PBS/Slingshot, scales by just changing `-n`.
**Cons:** Requires `mpi4py` or MPI-aware env vars in the script (handled
by `utils/distributed.py`).

### Summary

| Approach | Single Node | Multi-Node | Complexity |
|----------|:-----------:|:----------:|:----------:|
| `torchrun` | Yes | No (needs mpiexec) | Low |
| `mpiexec` + `torchrun` | Yes | Yes | Medium |
| `mpiexec` alone | Yes | Yes | Low |

> **Tip:** All PBS scripts in this repo use Option 3 (`mpiexec` alone).
> Each script is self-contained — copy any `.sh` file as a template for your
> own job. For interactive testing on a single GPU node, `torchrun --standalone`
> is the quickest way to get started.

## Derecho-Specific Configuration

Every PBS script should include these settings for Slingshot:

```bash
# NCCL
export NCCL_SOCKET_IFNAME=hsn    # Use Slingshot network (required)
export NCCL_IB_DISABLE=1         # Disable InfiniBand (required)
export NCCL_SHM_DISABLE=1        # Avoid occasional hangs
export NCCL_CROSS_NIC=1          # Better multi-NIC performance

# Libfabric CXI (Slingshot provider)
export FI_CXI_DISABLE_HOST_REGISTER=1   # Prevent CUDA deadlocks
export FI_CXI_DEFAULT_CQ_SIZE=131072    # Larger completion queue
```

> **Note:** By default, conda-installed PyTorch uses NCCL's **socket
> transport** over Slingshot. For better multi-node performance, you can
> use the native OFI transport via the
> [AWS OFI NCCL Plugin](https://github.com/aws/aws-ofi-nccl) — see
> [`docs/guide/nccl_tuning.md`](docs/guide/nccl_tuning.md) for details.

See any of the per-strategy PBS scripts (e.g., `run_fsdp.sh`) for a
complete, copy-paste-ready template.

## References

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [NCAR Derecho Documentation](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/)

---

<p align="center">
  <i>Questions? Open an issue or contact <a href="https://github.com/negin513">@negin513</a></i>
</p>
