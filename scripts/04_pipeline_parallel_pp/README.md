# Pipeline Parallelism (PP)

Pipeline parallelism splits a model **vertically by layers** across GPUs.
Each GPU ("stage") holds a subset of layers, and micro-batches flow through
the pipeline like an assembly line.

## Why Pipeline Parallelism?

When a model is too deep to fit on a single GPU (e.g., 96-layer transformer),
you can assign layers 0-23 to GPU 0, 24-47 to GPU 1, and so on. Each GPU
only holds 1/N of the parameters — enabling training of very deep models.

## The Pipeline Bubble Problem

Naive pipeline execution wastes GPU time. The key insight of PP research
is **scheduling** micro-batches to minimize idle time ("bubbles").

### GPipe Schedule

All micro-batches run forward first, then all run backward.
Simple but has a large bubble at the transition:

```
Time ──────────────────────────────────────────────────────────────►

GPU 0 (layers 0-1):  [F0][F1][F2][F3][  bubble  ][B3][B2][B1][B0]
GPU 1 (layers 2-3):  [  ][F0][F1][F2][F3][      ][B3][B2][B1][B0]
GPU 2 (layers 4-5):  [  ][  ][F0][F1][F2][F3][  ][B3][B2][B1][B0]
GPU 3 (layers 6-7):  [  ][  ][  ][F0][F1][F2][F3][B3][B2][B1][B0]
                                              ^^^^
                                           pipeline bubble
                                        (GPUs idle, wasted time)

F0 = forward micro-batch 0    B0 = backward micro-batch 0
```

**Bubble fraction:** (num_stages - 1) / num_microbatches

### 1F1B Schedule (One Forward, One Backward)

Interleaves forward and backward passes to keep GPUs busier:

```
Time ──────────────────────────────────────────────────────────────►

GPU 0:  [F0][F1][F2][F3][B0][B1][B2][B3]
GPU 1:  [  ][F0][F1][F2][B0][F3][B1][B2][B3]
GPU 2:  [  ][  ][F0][F1][B0][F2][B1][F3][B2][B3]
GPU 3:  [  ][  ][  ][F0][B0][F1][B1][F2][B2][F3][B3]
                       ▲
                 warmup fills pipe, then steady-state 1F1B
```

**Result:** Smaller bubble, lower peak memory (activations freed sooner).

### How a Model is Split Across GPUs

```
Original Model (8 layers):
┌──────────────────────────────────────────────────┐
│ Layer0 │ Layer1 │ Layer2 │ Layer3 │ ... │ Layer7 │
└──────────────────────────────────────────────────┘

After splitting across 4 GPUs:
┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│ GPU 0      │  │ GPU 1      │  │ GPU 2      │  │ GPU 3      │
│ Stage 0    │  │ Stage 1    │  │ Stage 2    │  │ Stage 3    │
│            │  │            │  │            │  │            │
│ Layer 0    │  │ Layer 2    │  │ Layer 4    │  │ Layer 6    │
│ Layer 1    │──│ Layer 3    │──│ Layer 5    │──│ Layer 7    │
│            │  │            │  │            │  │            │
│ Params:25% │  │ Params:25% │  │ Params:25% │  │ Params:25% │
└────────────┘  └────────────┘  └────────────┘  └────────────┘
       │               │               │               │
       └───── activations flow via send/recv ──────────┘
```

## When to Use Pipeline Parallelism

**Use PP when:**
- Model is too **deep** (many layers) for one GPU
- You have fast inter-GPU bandwidth (NVLink, Slingshot)
- Combined with TP for very large models (3D parallelism)

**Don't use PP when:**
- Model fits on one GPU — use DDP instead
- Model is wide but shallow — use TP instead
- You only have 1-2 GPUs — bubble overhead dominates

## Decision Guide

```
Model fits on 1 GPU?
├── Yes → Use DDP (simplest, best scaling)
└── No
    ├── Wide layers (large hidden dim)? → Use TP
    ├── Many layers (deep model)? → Use PP
    └── Both? → Use TP + PP (3D parallelism)
```

## Files in this Directory

| Script | GPUs | What it demonstrates |
|--------|------|---------------------|
| `01_manual_model_split.py` | 4 | Manual model splitting with send/recv — the fundamental concept |
| `02_pipeline_schedules.py` | 4 | PyTorch pipelining API with GPipe and 1F1B schedules |
| `03_pipeline_training.py` | 4 | Full training loop with PP on a small transformer |

## Running the Examples

### With mpiexec (recommended on Derecho)

```bash
mpiexec -n 4 --ppn 4 --cpu-bind none python 01_manual_model_split.py
mpiexec -n 4 --ppn 4 --cpu-bind none python 02_pipeline_schedules.py
mpiexec -n 4 --ppn 4 --cpu-bind none python 03_pipeline_training.py
```

### With torchrun (single node, 4 GPUs)

```bash
torchrun --standalone --nproc_per_node=4 01_manual_model_split.py
torchrun --standalone --nproc_per_node=4 02_pipeline_schedules.py
torchrun --standalone --nproc_per_node=4 03_pipeline_training.py
```

### Multi-node with mpiexec (2 nodes x 4 GPUs)

```bash
export NCCL_SOCKET_IFNAME=hsn
export NCCL_IB_DISABLE=1
export NCCL_CROSS_NIC=1
mpiexec -n 8 --ppn 4 --cpu-bind none python 03_pipeline_training.py --num-steps 20
```

### Via PBS on Derecho

```bash
qsub run_pipeline_parallel.sh
```

## Prerequisites

- PyTorch >= 2.4 (for `torch.distributed.pipelining`)
- 4 GPUs minimum

## References

- [GPipe Paper](https://arxiv.org/abs/1811.06965)
- [PipeDream / 1F1B Paper](https://arxiv.org/abs/1806.03377)
- [PyTorch Pipelining API](https://pytorch.org/docs/stable/distributed.pipelining.html)
