# Distributed PyTorch Training on NCAR's Derecho

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Derecho](https://img.shields.io/badge/HPC-Derecho-green)](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/)

**Developed by:** [Negin Sobhani](https://github.com/negin513) | CISL, NSF-NCAR

## Overview

This repository provides production-ready examples and templates for distributed PyTorch training on NCAR's [Derecho](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/) supercomputer (82 nodes x 4 A100 GPUs). While this code is written to run on Derecho GPU nodes, it can be adapted for other HPC machines.

The goal of this repository is to provide a starting point for researchers who want to scale their PyTorch training to multiple GPUs and nodes on NCAR's HPC systems.

Each strategy includes working code, PBS scripts, and ASCII diagrams explaining the concepts.

## Derecho GPU Resources

| Component | Specification |
|-----------|---------------|
| **GPU Nodes** | 82 nodes |
| **GPUs per Node** | 4x NVIDIA A100 (40 GB) |
| **Interconnect** | HPE Slingshot |
| **Scheduler** | PBS Pro |

**Peak Capability:** 82 nodes x 4 GPUs = **328 A100 GPUs**

## Strategy Comparison

| Strategy | Dir | What's Split | Communication | Memory Savings | Best For |
|----------|-----|-------------|---------------|----------------|----------|
| **DDP** | `01_*` | Data (batches) | Gradient all-reduce | None (full replica) | Most workloads |
| **FSDP** | `02_*` | Params + grads + optimizer | All-gather + reduce-scatter | High | Large models |
| **TP** | `03_*` | Weight matrices | All-reduce on activations | Medium | Wide layers (LLMs) |
| **PP** | `04_*` | Model layers | Send/recv between stages | High | Very deep models |
| **SP** | `05_*` | Sequence dimension | All-gather + reduce-scatter | Medium | Long sequences |
| **Hybrid** | `06_*` | TP + FSDP combined | Both | High | Multi-node LLMs |
| **Domain** | `07_*` | Spatial dimensions | Halo exchange (P2P) | High | High-res spatial data |

## Parallelism Strategies Explained

### What is DDP (Distributed Data Parallel)?

[Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) is PyTorch's most widely used distributed training strategy. Each GPU holds a **complete copy** of the model and processes a different slice of the training data. After each backward pass, gradients are synchronized across all GPUs using an all-reduce operation, ensuring every replica stays in sync. DDP scales your effective batch size linearly — 4 GPUs means 4x the data throughput. **Start here** if your model fits on a single GPU. It's the simplest strategy and often the fastest.

See [`scripts/01_data_parallel_ddp/`](scripts/01_data_parallel_ddp/) for examples.

### What is FSDP (Fully Sharded Data Parallelism)?

[Fully Sharded Data Parallel (FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) solves the main limitation of DDP: memory. Instead of keeping a full copy of the model on every GPU, FSDP **shards** the model parameters, gradients, and optimizer states across GPUs. Each GPU only stores 1/N of the total state (where N = number of GPUs). When a layer needs its full parameters (during forward/backward), FSDP temporarily all-gathers them from other GPUs, computes, then discards the non-local shards. This trades communication for memory, enabling models that are 4-8x larger than what DDP can handle.

See [`scripts/02_fully_sharded_fsdp/`](scripts/02_fully_sharded_fsdp/) for examples. For a deeper comparison of DDP vs FSDP, see [this article](https://openmmlab.medium.com/its-2023-is-pytorch-s-fsdp-the-best-choice-for-training-large-models-fe8d2848832f).

### What is Tensor Parallelism (TP)?

[Tensor Parallelism](https://pytorch.org/tutorials/intermediate/TP_tutorial.html) splits individual **weight matrices** across GPUs. Instead of each GPU having a full copy of a large linear layer, the weight matrix is divided column-wise or row-wise so each GPU computes a portion of the output. TP is most effective for models with very large individual layers (e.g., the attention and FFN layers in LLMs). It requires high-bandwidth GPU interconnect since activations are communicated at every layer. On Derecho, keep TP degree ≤ 4 (within a single node).

See [`scripts/03_tensor_parallel_tp/`](scripts/03_tensor_parallel_tp/) for examples.

### What is Pipeline Parallelism (PP)?

Pipeline Parallelism splits a model **by layers** across GPUs, forming a pipeline of stages. GPU 0 runs layers 0-9, GPU 1 runs layers 10-19, and so on. The training batch is split into microbatches that flow through the pipeline, allowing multiple GPUs to be active simultaneously. PP uses point-to-point send/recv (not all-reduce), so it has low communication overhead. The tradeoff is the pipeline bubble — stages sit idle while the pipeline fills and drains. More microbatches reduce the bubble. Best for very deep models (100+ layers).

See [`scripts/04_pipeline_parallel_pp/`](scripts/04_pipeline_parallel_pp/) for examples.

### What is Sequence Parallelism (SP)?

Sequence Parallelism splits the **sequence dimension** of activations across GPUs. In transformer models, the attention mechanism produces activations of shape `(batch, sequence_length, hidden_dim)`. When sequence lengths are very long, these activations dominate memory usage. SP distributes this cost across GPUs. SP is typically combined with TP — TP handles the attention/FFN computation while SP handles the LayerNorm and dropout operations on split sequences. This combination gives memory savings on both parameters and activations.

See [`scripts/05_sequence_parallel_sp/`](scripts/05_sequence_parallel_sp/) for examples.

### What is Hybrid Parallelism (TP + FSDP)?

For the largest models, a single strategy isn't enough. Hybrid parallelism combines TP within a node (where GPU bandwidth is highest) with FSDP across nodes (where communication cost is higher but less frequent). This is the standard approach for training LLMs at scale. PyTorch's `DeviceMesh` API makes it straightforward to define the 2D mesh of (TP, FSDP) dimensions.

See [`scripts/06_hybrid_parallelism/`](scripts/06_hybrid_parallelism/) for examples.

### What is Domain Parallelism?

Domain Parallelism splits the **spatial dimensions** of input data across GPUs. This is particularly relevant for climate/weather models and other scientific applications that process high-resolution spatial grids (e.g., 1024×1024 images or 3D atmospheric fields). Unlike DDP (which replicates the model), domain parallelism keeps the model on each GPU but splits the data spatially. Convolutions near partition boundaries require halo exchange — neighboring GPUs send each other the border rows/columns they need.

See [`scripts/07_domain_parallel_shardtensor/`](scripts/07_domain_parallel_shardtensor/) for examples.

## Repository Structure

```
scripts/
├── 01_data_parallel_ddp/          # Start here — simplest approach
│   ├── multinode_ddp_basic.py     #   Minimal DDP with synthetic data
│   ├── distributed_dataloader.py  #   DistributedSampler + DataLoader
│   └── torchrun_multigpu_ddp.sh   #   PBS job script
│
├── 02_fully_sharded_fsdp/         # When model > 40 GB
│   └── resnet_fsdp_training.py    #   FSDP training with ResNet-18
│
├── 03_tensor_parallel_tp/         # When individual layers are huge
│   ├── 01_basic_tensor_parallel.py
│   ├── 02_device_mesh_example.py
│   ├── 03_2d_tensor_parallel.py
│   └── 04_advanced_tp_example.py
│
├── 04_pipeline_parallel_pp/       # When model has 100+ layers
│   ├── 01_manual_model_split.py   #   Manual send/recv between stages
│   ├── 02_pipeline_schedules.py   #   GPipe vs 1F1B schedules
│   └── 03_pipeline_training.py    #   Full training loop with PP
│
├── 05_sequence_parallel_sp/       # Long sequences blowing up memory
│   ├── 01_basic_sequence_parallel.py
│   ├── 02_sp_transformer_layer.py
│   └── 03_sp_training.py
│
├── 06_hybrid_parallelism/         # TP within nodes + FSDP across nodes
│   ├── 01_fsdp_tp_hybrid.py
│   └── llama2_model.py
│
├── 07_domain_parallel_shardtensor/  # High-res spatial data (weather/climate)
│   ├── 01_basic_shardtensor.py
│   ├── 02_shardtensor_conv.py
│   ├── 03_domain_parallel_training.py
│   └── 04_domain_parallel_with_fsdp.py
│
├── pbs_common.sh                  # Shared PBS setup (source this)
│
utils/
├── distributed.py                 # Rank detection for all launchers
├── logging.py                     # Rank-aware logging
├── config.py                      # Training configuration
├── profiling.py                   # PyTorch profiler wrapper
└── checkpointing.py               # Save/load checkpoints

docs/
├── derecho_guide.md               # Derecho hardware and PBS reference
├── nccl_tuning.md                 # NCCL env vars for Slingshot
├── troubleshooting.md             # Common errors and fixes
└── strategy_decision_guide.md     # Choosing the right strategy
```

## Quick Start

### 1. Setup Environment

```bash
git clone https://github.com/NCAR/distributed-pytorch-hpc
cd distributed-pytorch-hpc

module load conda
conda env create -f environment.yml
conda activate pytorch-derecho
```

### 2. Verify Your Setup

```bash
# Check that PyTorch sees GPUs and NCCL is available
python -c "import torch; print(f'PyTorch {torch.__version__}, GPUs: {torch.cuda.device_count()}, NCCL: {torch.cuda.nccl.version()}')"
```

### 3. Run Your First Distributed Job

```bash
# Start with single-node multi-GPU (4 GPUs)
cd scripts/01_data_parallel_ddp
qsub torchrun_multigpu_ddp.sh

# Monitor your job
qstat -u $USER
watch -n 5 qstat -u $USER

# Check output
cat *.log
```

Or run interactively (from an interactive GPU node):

```bash
# With mpiexec (recommended on Derecho)
mpiexec -n 4 --ppn 4 --cpu-bind none python multinode_ddp_basic.py

# With torchrun
torchrun --standalone --nproc_per_node=4 multinode_ddp_basic.py
```

### 4. Scale to Multiple Nodes

Once single-node works, scale to 2 nodes (8 GPUs):

```bash
# Edit the PBS script: change select=1 to select=2, then:
qsub torchrun_multigpu_ddp.sh

# Or interactively:
mpiexec -n 8 --ppn 4 --cpu-bind none python multinode_ddp_basic.py
```

### 5. Try Other Strategies

Once DDP works, explore based on your needs:

```bash
# FSDP — model too large for 1 GPU
cd scripts/02_fully_sharded_fsdp
qsub run_fsdp.sh

# Tensor Parallelism — very large layers
cd scripts/03_tensor_parallel_tp
qsub run_tensor_parallel.sh

# Pipeline Parallelism — very deep model
cd scripts/04_pipeline_parallel_pp
qsub run_pipeline_parallel.sh

# Sequence Parallelism — long sequences
cd scripts/05_sequence_parallel_sp
qsub run_sequence_parallel.sh
```

## Launching Distributed Jobs

There are three ways to launch distributed PyTorch scripts. All examples in
this repo work with any of them — `utils/distributed.py` auto-detects the
launcher and sets up ranks accordingly.

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
`pbs_common.sh` and all PBS scripts in this repo.

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

> **Tip:** All PBS scripts in this repo use Option 3 (`mpiexec` alone) via
> the `launch_distributed()` helper in
> [`pbs_common.sh`](scripts/pbs_common.sh). For interactive testing on a
> single GPU node, `torchrun --standalone` is the quickest way to get started.

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
> [`docs/nccl_tuning.md`](docs/nccl_tuning.md) for details.

See [`scripts/pbs_common.sh`](scripts/pbs_common.sh) for the full setup.

## Documentation

| Guide | Description |
|-------|-------------|
| [Strategy Decision Guide](docs/strategy_decision_guide.md) | Choosing the right parallelism strategy |
| [Derecho Guide](docs/derecho_guide.md) | Hardware topology, PBS, and launch patterns |
| [NCCL Tuning](docs/nccl_tuning.md) | NCCL environment variables for Slingshot |
| [Troubleshooting](docs/troubleshooting.md) | Common errors and solutions |

## References

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053)
- [NCAR Derecho Documentation](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/)

---

<p align="center">
  <i>Questions? Open an issue or contact <a href="https://github.com/negin513">@negin513</a></i>
</p>
