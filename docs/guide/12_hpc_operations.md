# Chapter 12: HPC Operations

This chapter covers the practical side of running distributed PyTorch on
an HPC cluster, with Derecho-specific details throughout for PBSPro job
scripts, launch methods, NCCL tuning, and debugging.


Topics include:
- PBS job scripts
- Launch methods: `torchrun`, `mpiexec + python`, `mpiexec + torchrun`
- NCCL tuning for Slingshot
- Debugging distributed jobs

## PBS Basics

PBS Pro (Portable Batch System) is Derecho's job scheduler. 

You write a shell script with `#PBS` directives and submit it with `qsub`.

### Minimal GPU Job Script

```bash
#!/bin/bash
#PBS -A YOUR_PROJECT                           # allocation code
#PBS -N my_training                            # job name
#PBS -l walltime=01:00:00                      # max runtime
#PBS -l select=1:ncpus=64:ngpus=4:mem=480GB    # 1 node, 4 GPUs
#PBS -q main                                   # GPU queue
#PBS -j oe                                     # merge stdout + stderr

# --- Environment ---
module purge
module load nvhpc cuda cray-mpich conda
conda activate pytorch-derecho

cd $PBS_O_WORKDIR

# --- Launch ---
torchrun --standalone --nproc_per_node=4 train.py
```

### Multi-Node Job

```bash
#PBS -l select=2:ncpus=64:ngpus=4:mem=480GB   # 2 nodes = 8 GPUs
```
and update to use `mpiexec` instead:

```
NNODES=$(cat $PBS_NODEFILE | sort -u | wc -l)
NGPUS=$((NNODES * 4))

mpiexec -n $NGPUS --ppn 4 --cpu-bind none python train.py
```


### Key PBS Directives

| Directive | Meaning |
|-----------|---------|
| `#PBS -A PROJ` | Charge hours to this project |
| `#PBS -l select=N` | Number of nodes |
| `#PBS -l ncpus=64` | CPU cores per node (Derecho: always 64) |
| `#PBS -l ngpus=4` | GPUs per node (Derecho: always 4) |
| `#PBS -l mem=480GB` | RAM per node (max ~480 GB usable) |
| `#PBS -l walltime=HH:MM:SS` | Maximum wall clock time |
| `#PBS -q main` | GPU queue |
| `#PBS -j oe` | Combine stdout and stderr into one file |

### PBS Commands

```bash
qsub script.sh           # submit a job
qstat -u $USER           # check your jobs
qdel JOB_ID              # cancel a job
qstat -f JOB_ID          # full job details
```


## Three Launch Methods

### 1. torchrun (single node)

Best for single-node jobs. Handles rank assignment and master addr/port
automatically:

```bash
torchrun --standalone --nproc_per_node=4 train.py
```

- Sets `LOCAL_RANK`, `RANK`, `WORLD_SIZE` environment variables
- Your script reads these to initialize the process group

### 2. mpiexec + Python (multi-node -- recommended).

Uses MPI to launch one process per GPU. Your script must detect ranks
from MPI environment variables or use `mpi4py`:

```bash
NNODES=$(cat $PBS_NODEFILE | sort -u | wc -l)
NGPUS=$((NNODES * 4))

mpiexec -n $NGPUS --ppn 4 --cpu-bind none python train.py
```

- `--ppn 4`: 4 processes per node
- `--cpu-bind none`: don't pin to specific CPU cores
- Your script detects `OMPI_COMM_WORLD_RANK` or uses `mpi4py`

### 3. mpiexec + torchrun (multi-node)

Combines MPI for node discovery with torchrun for per-node process
management. Best for multi-node jobs:

```bash
NNODES=$(cat $PBS_NODEFILE | sort -u | wc -l)
HEAD_NODE=$(head -1 $PBS_NODEFILE)
HEAD_ADDR=$(hostname -i)

mpiexec -n $NNODES --ppn 1 \
    torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$HEAD_ADDR:29500 \
    train.py
```

- MPI launches one torchrun per node
- torchrun spawns 4 workers per node
- Rendezvous via c10d on the head node

### Comparison

| Method | Nodes | Rank Detection | Complexity |
|--------|-------|---------------|------------|
| `torchrun --standalone` | 1 | Automatic | Simple |
| `mpiexec + python` | 1+ | Manual (env vars / mpi4py) | Medium |
| `mpiexec + torchrun` | 1+ | Error-prone | Medium |

## NCCL Tuning for Slingshot

NCCL (NVIDIA Collective Communications Library) handles GPU-to-GPU
communication. On Derecho's Slingshot network, you need specific settings to ensure it uses the correct transport and interfaces.

The default NCCL configuration may not work correctly on Slingshot, leading to hangs or errors. You must set the following environment variables:

```bash
export NCCL_SOCKET_IFNAME=hsn          # use Slingshot interfaces
export NCCL_IB_DISABLE=1               # no InfiniBand on Derecho
```

Without these, NCCL will try to use loopback or non-existent InfiniBand
and either hang or crash.

AWS_OFI (OpenFabrics Interfaces) is the underlying transport for Slingshot. You can further optimize NCCL with OFI-specific settings:


```bash
### Recommended


export NCCL_SHM_DISABLE=1              # disable shared memory (stability)
export NCCL_CROSS_NIC=1                # enable cross-NIC communication
export CUDA_VISIBLE_DEVICES=0,1,2,3    # expose all 4 GPUs
```

### Libfabric / CXI (Slingshot transport)

```bash
export FI_CXI_DISABLE_HOST_REGISTER=1  # prevent CUDA deadlocks
export FI_CXI_DEFAULT_CQ_SIZE=131072   # larger completion queue for big jobs
export FI_MR_CACHE_MONITOR=userfaultfd # memory registration cache
```

### All together

Every PBS script in `scripts/` includes these. Here's the full block:

```bash
# NCCL settings for Derecho
export NCCL_SOCKET_IFNAME=hsn
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_CROSS_NIC=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Libfabric / CXI for Slingshot
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd
```

For the full NCCL reference including OFI transport and advanced tuning,
see [`nccl_tuning.md`](nccl_tuning.md).

## Debugging Distributed Jobs

### Step 1: Check the basics

Most distributed failures come from a few common issues:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Hangs at init | Wrong network interface | Set `NCCL_SOCKET_IFNAME=hsn` |
| Hangs at init | Firewall / port blocked | Try different `MASTER_PORT` |
| `NCCL error: unhandled system error` | Missing NCCL config | Add all NCCL exports |
| `CUDA error: invalid device ordinal` | Wrong GPU count | Check `CUDA_VISIBLE_DEVICES` |
| `RuntimeError: address already in use` | Port conflict | Change port or wait for old job to end |
| Incorrect results | Ranks out of sync | Check `set_epoch()` on sampler |
| OOM on rank 0 only | Data download on rank 0 | Use barrier after download |

### Step 2: Enable debug logging

```bash
# NCCL debug output (shows connection setup)
export NCCL_DEBUG=INFO

# Very verbose (for deep debugging)
export NCCL_DEBUG=TRACE

# PyTorch distributed debug (shows collective operations)
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

### Step 3: Test communication in isolation

Before debugging your training script, verify that GPUs can talk at all:

```bash
# All-reduce test
mpiexec -n 4 --ppn 4 --cpu-bind none python tests/all_reduce_test.py

# Point-to-point test
mpiexec -n 4 --ppn 4 --cpu-bind none python tests/send_recv_test.py

# Full benchmark
mpiexec -n 4 --ppn 4 --cpu-bind none python tests/torch_comm_bench.py
```

If these fail, the problem is in the environment, not your code.

### Step 4: Check environment

```bash
# Verify GPU visibility
python -c "import torch; print(torch.cuda.device_count(), 'GPUs')"

# Verify NCCL version
python -c "import torch; print('NCCL:', torch.cuda.nccl.version())"

# Run the environment check script
python tests/check_environment.py
```

### Common Debugging Patterns

**Hang during training (not init):**
Usually a rank mismatch — one rank takes a different code path (e.g.,
different batch count due to uneven data). Ensure all ranks execute the
same number of collective operations.

**OOM during training:**
- Check batch size (effective = per_gpu × world_size)
- Try mixed precision (`--use-amp`)
- Switch from DDP to FSDP
- Reduce micro-batch size for PP

**Slow performance:**
- Check NCCL transport: `NCCL_DEBUG=INFO` should show `ofi` or `socket`
- Verify GPU utilization: `nvidia-smi` during training
- Profile with `torch.profiler` (see `utils/profiling.py`)

## Putting It All Together

Here's a complete multi-node PBS template that incorporates everything:

```bash
#!/bin/bash
#PBS -A YOUR_PROJECT
#PBS -N distributed_training
#PBS -l walltime=02:00:00
#PBS -l select=2:ncpus=64:ngpus=4:mem=480GB
#PBS -q main
#PBS -j oe

# --- Environment ---
module purge
module load nvhpc cuda cray-mpich conda
conda activate pytorch-derecho

# --- NCCL ---
export NCCL_SOCKET_IFNAME=hsn
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_CROSS_NIC=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd

# --- Launch ---
cd $PBS_O_WORKDIR

NNODES=$(cat $PBS_NODEFILE | sort -u | wc -l)
NGPUS=$((NNODES * 4))

mpiexec -n $NGPUS --ppn 4 --cpu-bind none \
    python your_training_script.py \
    --batch-size 64 \
    --epochs 10
```

Copy any `.sh` file from `scripts/` as a starting point — they all
follow this pattern.

---

**Deep-dive references:**
- [`derecho_guide.md`](derecho_guide.md) — full hardware specs, PBS reference, launch patterns
- [`nccl_tuning.md`](nccl_tuning.md) — OFI transport, advanced NCCL settings
- [`troubleshooting.md`](troubleshooting.md) — comprehensive error catalog with solutions

---

**This concludes the guide.** You now have the conceptual foundations and
practical tools to run distributed PyTorch training — from a single GPU
to hundreds. Return to the [table of contents](README.md) to review any
chapter, or explore the scripts in each strategy directory for working
code.
