# Derecho GPU Cluster Guide for Distributed PyTorch

## Hardware Topology

```
NCAR Derecho — GPU Partition
═══════════════════════════════════════════════════════════════

82 GPU Nodes, each:
  Single-socket, 64 cores per socket
  512 GB DDR4 memory per node
┌──────────────────────────────────────────────────────┐
│  Node (AMD EPYC 7763 — 64 cores, 512 GB RAM)         │
│                                                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │ A100    │  │ A100    │  │ A100    │  │ A100    │  │
│  │ 40 GB   │  │ 40 GB   │  │ 40 GB   │  │ 40 GB   │  │
│  │ HBM2    │  │ HBM2    │  │ HBM2    │  │ HBM2    │  │
│  │ GPU 0   │  │ GPU 1   │  │ GPU 2   │  │ GPU 3   │  │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  │
│       │            │            │            │       │
│       └──── NVLink (600 GB/s) ──────────────┘        │
│                                                      │
└──────────────────────┬───────────────────────────────┘
                       │
                 HPE Slingshot 11
                 (200 Gbps per node)
                       │
┌──────────────────────┴───────────────────────────────┐
│                Slingshot Fabric                      │
│         (Dragonfly topology, 82 GPU nodes)           │
└──────────────────────────────────────────────────────┘

Peak: 82 nodes × 4 GPUs = 328 A100 GPUs
```

## PBS Job Script Reference

### Minimal GPU Job

```bash
#!/bin/bash
#PBS -A YOUR_PROJECT    
#PBS -N job_name        # Job name (shows in qstat)
#PBS -l walltime=01:00:00                          
#PBS -l select=1:mpiprocs=1:ncpus=64:ngpus=4
#PBS -q main            # Queue (main for GPU jobs)
#PBS -j oe              # Merge stdout and stderr
```

| Directive | Meaning |
|-----------|---------|
| `#PBS -A PROJ` | Charge hours to this project |
| `#PBS -l select=N` | Number of nodes |
| `#PBS -l mpiprocs=M` | MPI ranks per node |
| `#PBS -l ncpus=64` | CPU cores per node (64 on Derecho) |
| `#PBS -l ngpus=4` | GPUs per node (4 on Derecho) |
| `#PBS -l mem=480GB` | RAM per node (max ~480 GB usable) |
| `#PBS -j oe` | Combine stdout/stderr into one file |
| `#PBS -l walltime=HH:MM:SS` | Maximum wall clock time |

### Multi-Node Example

```bash
#PBS -l select=2:ncpus=64:ngpus=4:mem=480GB
# This requests 2 nodes = 8 GPUs total
```

### Useful PBS Commands

```bash
qsub script.sh          # Submit a job
qstat -u $USER           # Check your jobs
qdel JOB_ID              # Cancel a job
qstat -f JOB_ID          # Full job details
qstat -Q main            # Queue status
```

## Module Loading and Conda Setup

```bash
# Standard module stack for distributed PyTorch
module purge
module load nvhpc cuda cray-mpich conda

# Activate your conda environment
conda activate pytorch-derecho
```

### Creating the Conda Environment

```bash
module load conda
conda env create -f environment.yml
conda activate pytorch-derecho

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Launch Patterns

### Single Node (4 GPUs)

```bash
# Option 1: mpiexec (recommended on Derecho)
mpiexec -n 4 --ppn 4 --cpu-bind none python train.py

# Option 2: torchrun
torchrun --standalone --nproc_per_node=4 train.py
```

### Multi-Node (2 nodes, 8 GPUs)

```bash
# mpiexec launches one process per GPU across all nodes
mpiexec -n $((NNODES * 4)) --ppn 4 --cpu-bind none python train.py
```

Each PBS script in `scripts/` is a self-contained template with all
required modules, NCCL config, and node discovery inlined — copy any
`.sh` file and adjust the `#PBS` directives and `python` command for
your job.

## Common Environment Variables

### Required for Derecho

```bash
export NCCL_SOCKET_IFNAME=hsn        # Use Slingshot interfaces
export NCCL_IB_DISABLE=1             # No InfiniBand on Derecho
```

### Recommended

```bash
export NCCL_SHM_DISABLE=1            # Disable shared memory (stability)
export NCCL_CROSS_NIC=1              # Enable cross-NIC communication
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Expose all 4 GPUs
```

#### Libfabric / CXI (Slingshot)

```bash
export FI_CXI_DISABLE_HOST_REGISTER=1   # Prevent CUDA deadlocks with CXI
export FI_CXI_DEFAULT_CQ_SIZE=131072    # Larger completion queue for big jobs
export FI_MR_CACHE_MONITOR=userfaultfd  # Memory registration cache monitor
```

### For Debugging

```bash
export NCCL_DEBUG=INFO               # Verbose NCCL logging
export NCCL_DEBUG=TRACE              # Very verbose (generates lots of output)
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # PyTorch distributed debugging
```

See [`docs/nccl_tuning.md`](nccl_tuning.md) for the full NCCL configuration
reference, including how to use the native OFI transport for better multi-node
performance.
