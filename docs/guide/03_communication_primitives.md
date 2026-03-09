# Chapter 3: Communication Primitives (Collective Operations)

Every distributed strategy from [Chapter 2](02_why_distributed.md) boils down to a pattern of
communication (collective operations) between GPUs. 

This chapter covers the building blocks:
ranks, process groups, and the five collective operations you'll see
throughout the guide.

**Key insight:** Once you understand these primitives, every distributed
strategy is just a specific pattern of these operations.

## Processes and Ranks

Distributed training runs multiple copies of your training script simultaneously — one per GPU. Each copy is called a **process**, and each process has a unique identifier called a **rank**.

Each process is identified by three numbers (assigned by the launcher):

| Term | Meaning | Range |
|------|---------|-------|
| `WORLD_SIZE` | Total number of processes | — |
| `WORLD_RANK` (or `RANK`) | Global ID of this process | 0 to WORLD_SIZE-1 |
| `LOCAL_RANK` | ID within this node | 0 to GPUs_per_node-1 |


For example, with 2 nodes and 4 GPUs each, you have 8 processes with `WORLD_RANK`s 0-7. Each node has local `LOCAL_RANK`s 0-3. The `LOCAL_RANK` is used to assign a GPU to each process, while the `WORLD_RANK` is used for coordination (e.g., rank 0 handles logging and checkpointing). 
`WORLD_SIZE` is the total number of processes (8 in this example) and is used for scaling learning rates and calculating effective batch sizes.

```
┌─────────────────────────────────────────────────────────────────────┐
│              WORLD RANK VS LOCAL RANK (2 nodes × 4 GPUs)            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Node 0                              Node 1                        │
│   ┌───────────────────────────┐       ┌───────────────────────────┐ │
│   │  ┌─────┐ ┌─────┐          │       │  ┌─────┐ ┌─────┐          │ │
│   │  │GPU 0│ │GPU 1│          │       │  │GPU 0│ │GPU 1│          │ │
│   │  │     │ │     │          │       │  │     │ │     │          │ │
│   │  │LR=0 │ │LR=1 │          │       │  │LR=0 │ │LR=1 │          │ │
│   │  │WR=0 │ │WR=1 │          │       │  │WR=4 │ │WR=5 │          │ │
│   │  └─────┘ └─────┘          │       │  └─────┘ └─────┘          │ │
│   │  ┌─────┐ ┌─────┐          │       │  ┌─────┐ ┌─────┐          │ │
│   │  │GPU 2│ │GPU 3│          │       │  │GPU 2│ │GPU 3│          │ │
│   │  │     │ │     │          │       │  │     │ │     │          │ │
│   │  │LR=2 │ │LR=3 │          │       │  │LR=2 │ │LR=3 │          │ │
│   │  │WR=2 │ │WR=3 │          │       │  │WR=6 │ │WR=7 │          │ │
│   │  └─────┘ └─────┘          │       │  └─────┘ └─────┘          │ │
│   └───────────────────────────┘       └───────────────────────────┘ │
│                                                                     │
│   LR = LOCAL_RANK (0-3 on each node) → Use for: torch.cuda.set_device│
│   WR = WORLD_RANK (0-7 globally)     → Use for: distributed ops     │
│                                                                     │
│   WORLD_SIZE = 8 (total processes)                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

* You use `LOCAL_RANK` to assign a GPU: `torch.cuda.set_device(LOCAL_RANK)`.
* You use `WORLD_RANK` for coordination: rank 0 typically handles logging,
checkpointing, and data download.
* `WORLD_SIZE` is used for scaling learning rates and calculating effective batch sizes.
effective batch size = `batch_size × WORLD_SIZE`.


The above three variables are set by your launcher (e.g., `torchrun`, `mpiexec`) when you start your distributed job. The launcher ensures that each process gets the correct rank and world size, allowing them to coordinate properly.

## Launchers

To start a distributed training job, you use a launcher that spawns multiple processes with the appropriate environment variables set.

 The most common launchers are:

| Launcher | Use for | Example Command |
|----------|---------|-----------------|
| `torchrun` | Single-node multi-GPU | `torchrun --nproc_per_node=4 train.py` |
| `mpiexec` or `mpirun` | Multi-node | `mpiexec -n 8 --ppn 4 --cpu-bind none python train.py` |

 For multi-node training, `mpiexec` or `mpirun` can be used, but you need to ensure that the environment variables are correctly set across nodes.

You can also use `mpiexec` + `torchrun` for multi-node training, but this requires additional setup to ensure that the environment variables are correctly propagated across nodes.


### `torchrun`
`torchrun` is the recommended launcher for single-node multi-GPU training. It automatically sets `LOCAL_RANK`, `WORLD_RANK`, and `WORLD_SIZE` for each process.

For example, if you have 4 GPUs on a single node, you can launch your training script with:

```bash
torchrun --nproc_per_node=4 train.py
```
This will start 4 processes, each with `LOCAL_RANK` 0-3 and `WORLD_RANK` 0-3, and `WORLD_SIZE` 4.

For multi-node training, using `torchrun` requires additional setup to ensure that the environment variables are correctly propagated across nodes. This can be done with the `--rdzv_backend` and `--rdzv_endpoint` options, but it is often simpler to use `mpiexec` for multi-node training.

Essentially, you have to run `torchrun` on each node, and ensure that the `WORLD_RANK` and `WORLD_SIZE` are correctly set across nodes. This can be complex, which is why `mpiexec` is often preferred for multi-node training.

```
# On Node 1 -- (the master node)
torhcrun --nodes 2
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=node1 \ # IP or hostname of the master node
    --master_port=29500 \
    train.py    

# On Node 2 -- (the worker node)
torchrun --nodes 2
    --nproc_per_node=4 \
    --node_rank=1 \
    --master_addr=node1 \ # IP or hostname of the master node
    --master_port=29500 \
    train.py
```

In practice, for multi-node training, it's often simpler to use `mpiexec` or `mpirun`, which handles the multi-node coordination for you.

```
mpiexec -n 8 --ppn 4 --cpu-bind none python train.py
``` 

But you need to ensure that your training script correctly initializes the process group and sets the device based on the environment variables set by `mpiexec`.

The `utils/distributed.py` file in this repository provides a `setup_distributed()` function that abstracts away the details of initializing the process group and setting the device, making it easier to write distributed training scripts that can work with different launchers, such as `torchrun`, OpenMPI, or Cray MPICH.

This table summarizes the launchers and their environment variable handling:


| Launcher | WORLD_RANK | LOCAL_RANK | WORLD_SIZE |
|----------|------|------------|------------|
| torchrun | `RANK` | `LOCAL_RANK` | `WORLD_SIZE` |
| OpenMPI | `OMPI_COMM_WORLD_RANK` | `OMPI_COMM_WORLD_LOCAL_RANK` | `OMPI_COMM_WORLD_SIZE` |
| SLURM | `SLURM_PROCID` | `SLURM_LOCALID` | `SLURM_NTASKS` |
| Cray MPICH | `PMI_RANK` | `PMI_LOCAL_RANK` | `PMI_SIZE` |


## Process Groups

A **process group** is a set of ranks that communicate together. When you
call `init_process_group()`, it creates the default group containing all
ranks:

```python
import torch.distributed as dist

dist.init_process_group(backend="nccl")  # all ranks join

# All ranks can now communicate with each other
dist.all_reduce(tensor)  # works across all ranks
```

Sometimes you need only some ranks to communicate. This is common in hybrid parallelism (e.g., FSDP within nodes, DDP across nodes). In that case, you create **subgroups**:

```python
# Create a subgroup for ranks 0-3 (e.g., for TP within nodee 1)
tp_group = dist.new_group(ranks=[0, 1, 2, 3])

# Create a subgroup for ranks 4-7 (e.g., for TP within node 2)
tp_group_2 = dist.new_group(ranks=[4, 5, 6, 7])
```


### Backends

A **backend** is the library that actually performs the communication between GPUs. Different backends are optimized for different hardware.


| Backend | Hardware | Use for |
|---------|----------|---------|
| `nccl` | NVIDIA GPUs | GPU tensors (default for training) |
| `gloo` | CPU | CPU tensors, fallback |
| `mpi` | Any | When MPI is your launcher |

On Derecho, use `nccl` for GPU communication; but you need to ensure that your cluster has the necessary libraries installed and configured for Cray HPE Slingshot 11.
The wheel provided in `environment.yaml` includes NCCL support for Slingshot 11, but you may need to verify that the correct version of NCCL is being used and that it is properly configured for your cluster's network topology.



## The Five Collective Operations

**Collective operations** (or "collectives") are communication patterns where multiple ranks participate together. Every distributed training strategy is built from these primitives.

### 1. All-Reduce

Every GPU starts with a value. After all-reduce operations, every GPU has the
**sum** (or average, min, max) of all values across GPUs.

This is the core of DDP, where gradients are all-reduced after each backward pass to compute the average gradient across all GPUs before the optimizer step.

+![NVIDIA All-Reduce illustration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/allreduce.png)
Image source: NVIDIA NCCL documentation


### 2. All-Gather

Each GPU has a piece. After all-gather, every GPU has **all the pieces
concatenated**. 

FSDP uses this to reassemble sharded parameters before
forward/backward.


![NVIDIA All-Gather illustration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/allgather.png)

```
Before:           Operation:         After:
GPU 0: [A]                          GPU 0: [A, B, C, D]
GPU 1: [B]     ── all-gather ──►    GPU 1: [A, B, C, D]
GPU 2: [C]                          GPU 2: [A, B, C, D]
GPU 3: [D]                          GPU 3: [A, B, C, D]
```



### 3. Reduce-Scatter

The inverse of all-gather. Reduces (sums) data and **scatters** the
result so each GPU gets one piece. FSDP uses this after backward to
produce sharded gradients.

![NVIDIA Reduce-Scatter illustration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/reducescatter.png)

```
Before:                    Operation:              After:
GPU 0: [a0, a1, a2, a3]                          GPU 0: [a0+b0+c0+d0]
GPU 1: [b0, b1, b2, b3]  ── reduce-scatter ──►   GPU 1: [a1+b1+c1+d1]
GPU 2: [c0, c1, c2, c3]      (sum)               GPU 2: [a2+b2+c2+d2]
GPU 3: [d0, d1, d2, d3]                          GPU 3: [a3+b3+c3+d3]
```

!!! note: The all-gather + reduce-scatter pair is a powerful pattern for sharding data across GPUs while still allowing for global operations. FSDP uses this pattern to shard parameters and gradients, while TP can use it to shard activations.
![FSDP All-Gather illustration](https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-graph-2a.png)
Image source: Facebook FSDP paper


### 4. Broadcast

The Broadcast operation copies an N-element buffer from the root rank to all the ranks.
So one GPU sends its data to all others as shown in the image below....

Used for syncing initial model weights or distributing hyperparameters.

![NVIDIA Broadcast illustration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/broadcast.png)

```
Before:           Operation:         After:
GPU 0: [X]                          GPU 0: [X]
GPU 1: [ ]     ── broadcast ──►    GPU 1: [X]
GPU 2: [ ]        (from 0)         GPU 2: [X]
GPU 3: [ ]                          GPU 3: [X]
```


### 5. Point-to-Point (Send/Recv)

Direct communication between two specific GPUs. Pipeline parallelism
uses this: stage N sends activations to stage N+1.


### 6. All-to-All
In All-to-All, each GPU transmits unique data to every other GPU. To achieve this, each GPU breaks down its data into chunks. Subsequently, each GPU directly sends and receives these chunks to and from every other GPU. Finally, each GPU reconstructs the received data chunks.

![All-to-All illustration](https://miro.medium.com/v2/resize:fit:720/format:webp/1*k7blVUX0r9nb51Kg825Veg.png)

## Hardware Topology Matters

The speed of these collective operations depends entirely on the hardware connecting your GPUs. On virtually all modern supercomputers, **intra-node communication** (GPUs within the same server, connected via NVLink) is orders of magnitude faster than **inter-node communication** (servers connected via network fabrics like InfiniBand or Slingshot). 

This physical reality dictates how you map your distributed strategy to the hardware.

For example, on NCAR's Derecho system (which has 4× A100 GPUs per node):

* **Within a node (High Bandwidth):** Use communication-heavy, latency-sensitive strategies like Tensor Parallelism (TP) or Sequence Parallelism. These require constant, blocking data transfers during the forward and backward passes and must utilize NVLink.
* **Across nodes (Lower Bandwidth):** Use bandwidth-efficient strategies like FSDP or DDP. These strategies are designed to hide network latency by overlapping their communication (like gradient synchronization) with the ongoing computation of the backward pass.

See the [Derecho Guide](derecho_guide.md) for full hardware specs.

## Hands-On: Try the Primitives

The `tests/` directory has scripts that let you run these operations
directly:

- [`tests/all_reduce_test.py`](../../tests/all_reduce_test.py) — run an all-reduce and verify the result
- [`tests/send_recv_test.py`](../../tests/send_recv_test.py) — point-to-point communication between ranks
- [`tests/torch_comm_bench.py`](../../tests/torch_comm_bench.py) — benchmark all-reduce, broadcast, and send/recv at various tensor sizes

```bash
# Run on 4 GPUs
mpiexec -n 4 --ppn 4 --cpu-bind none python tests/all_reduce_test.py
```

## What's Next?

Now that you understand the communication building blocks, Chapter 4
puts them to use with the simplest and most common distributed strategy:
Data Parallel (DDP).

**Next:** [Chapter 4 — Data Parallel (DDP)](04_data_parallel_ddp.md)
