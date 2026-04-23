# Tensor Parallelism (TP)

Tensor Parallelism (TP) is a model-parallel partitioning method that distributes the parameter tensor of an individual layer across GPUs. In addition to reducing model state memory usage, it also saves activation memory as the per-GPU tensor sizes shrink. However, the reduced per-GPU tensor size increases CPU overhead due to smaller per-GPU kernel workloads.

![Tensor Parallelism Overview](../../docs/images/tp_overview.png)
*Figure 1: Tensor Parallelism distributes individual layer parameters across multiple GPUs.*

## How It Works

TP partitions large weight matrices across GPUs. For a linear layer `Y = XW`, there are two fundamental approaches:

### Column-Parallel Linear

The weight matrix is split along columns across GPUs. Each GPU receives an identical copy of the input and performs matrix multiplication on its column shard. The partial outputs are then concatenated via an all-gather operation.

![Column-wise Parallel](../../docs/images/tp_colwise.jpeg)
*Figure 2: Column-wise parallel splits the weight matrix W along columns. Each GPU computes a partial output, then results are gathered.*

### Row-Parallel Linear

The weight matrix is split along rows across GPUs. The input is divided along the inner dimension so each GPU has a corresponding shard. Each GPU computes a partial result, and outputs are combined via an all-reduce summation.

![Row-wise Parallel](../../docs/images/tp_rowwise.jpeg)
*Figure 3: Row-wise parallel splits the weight matrix W along rows. Each GPU computes a partial sum, then results are reduced.*

### Combined Column + Row Parallelism

In practice, sequential linear layers (e.g., in an MLP block) use both methods together. The column-wise output feeds directly into the row-wise layer **without any data transfer between GPUs**. Element-wise operations like activation functions also apply without communication overhead. This is the key insight from the [Megatron-LM paper](https://arxiv.org/abs/1909.08053).

![Combined Column and Row Parallel](../../docs/images/tp_combined.jpeg)
*Figure 4: Combined approach pairs column-wise and row-wise parallelism to minimize communication to a single all-reduce per block.*

## Files in this Directory

### `01_device_mesh_basics.py`
Introduction to PyTorch's `DeviceMesh` — the foundation for all parallelism strategies.

**What it covers:**
- Creating 1D meshes (flat GPU group)
- Creating 2D meshes (TP x DP layout)
- Slicing sub-meshes for passing to `parallelize_module()` or FSDP
- How mesh dimension ordering maps GPUs to nodes

### `02_basic_tensor_parallel.py`
Megatron-style tensor parallelism on a toy MLP using a 1D mesh.

**What it covers:**
- `ColwiseParallel` — split weight columns, all-gather output
- `RowwiseParallel` — split weight rows, all-reduce output
- Pairing them to minimize communication (one all-reduce per block)
- Full training loop: forward, backward, optimizer step

### `tensor_parallel_vit.py`
Full training example: a Vision Transformer (ViT) on synthetic ERA5-like weather data with tensor parallelism on a 2D mesh.

**What it covers:**
- Realistic model architecture (ViT with patch embedding, multi-head attention, MLP blocks)
- Megatron-LM style TP: Q/K/V projections (ColwiseParallel) + output projection (RowwiseParallel)
- ERA5-like dataset with latitude-weighted MSE loss (same as DDP and FSDP examples)
- Full training loop with throughput measurement
- 2D DeviceMesh: TP within nodes, DP across nodes
- Why TP is natural for Transformers (linear layers dominate, Conv2d models like UNet/ResNet are not TP-friendly)

## Running the Examples

### Submit all examples as a PBS job
```bash
qsub run_tensor_parallel.sh
```

### Run individually (single node, 4 GPUs)
```bash
# DeviceMesh basics
torchrun --standalone --nproc_per_node=4 01_device_mesh_basics.py
mpiexec -n 4 --ppn 4 --cpu-bind none python 01_device_mesh_basics.py

# Basic TP (1D mesh)
torchrun --standalone --nproc_per_node=4 02_basic_tensor_parallel.py
mpiexec -n 4 --ppn 4 --cpu-bind none python 02_basic_tensor_parallel.py

# TP ViT training (2D mesh, realistic model)
mpiexec -n 4 --ppn 4 --cpu-bind none python tensor_parallel_vit.py
mpiexec -n 8 --ppn 4 --cpu-bind none python tensor_parallel_vit.py  # 2 nodes
```

## Key Concepts

### Parallelization Primitives

PyTorch provides `torch.distributed.tensor.parallel` for TP:

```python
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

# Parallelize a linear layer column-wise
parallelize_module(
    module,
    device_mesh,
    {"linear": ColwiseParallel()}
)
```

### DeviceMesh

Organize GPUs into logical topologies:

```python
from torch.distributed.device_mesh import init_device_mesh

# 1D mesh for simple TP
mesh_1d = init_device_mesh("cuda", (4,))

# 2D mesh for TP + DP (last dim is fastest-varying = same node)
mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))
```

### Communication Patterns

TP requires specific communication patterns:

- **All-Gather**: Collect tensor shards from all GPUs (after ColwiseParallel)
- **All-Reduce**: Sum partial results across GPUs (after RowwiseParallel)
- **Reduce-Scatter**: Reduce and distribute results (used in some advanced layouts)

## Performance Tips

1. **Keep TP within a node** — TP requires high-bandwidth all-reduce; use NVLink, not network
2. **Use DP across nodes** — gradient all-reduce is more tolerant of network latency
3. **Ensure dimensions are divisible** — hidden_dim must be divisible by TP degree
4. **Pair ColwiseParallel + RowwiseParallel** — eliminates one communication step per block
5. **Profile with `NCCL_DEBUG=INFO`** — verify NCCL is using the expected transport

## When to Use Tensor Parallelism

**Good for:**
- Very large layers that don't fit on a single GPU
- Transformer models with large hidden dimensions
- Models with large embedding tables
- Combined with DP or FSDP for production training

**Not ideal for:**
- Small models (communication overhead dominates)
- GPUs connected via slow interconnect (TP needs high bandwidth)
- As the only strategy for medium-sized models (FSDP is usually better)

## Comparison with Other Strategies

| Strategy | Model Size | Communication | Memory Efficiency |
|----------|-----------|---------------|-------------------|
| DDP | Replicated | Gradients only | Low (full replica) |
| FSDP | Sharded | Parameters + gradients | High |
| TP | Distributed | Activations + gradients | Medium-High |

## Common Issues

### High Communication Overhead
- Ensure GPUs are connected via NVLink or high-speed interconnect
- Reduce TP degree (e.g., tp=2 instead of tp=4)
- Keep TP within a single node

### Incorrect Tensor Shapes
- Layer dimensions must be divisible by TP degree
- Check: `hidden_dim % tp == 0`
- Adjust model architecture or TP degree accordingly

## Combining with Other Strategies

TP is often combined with:

- **TP + DP**: Tensor parallel within nodes, data parallel across nodes (example 03)
- **TP + FSDP**: TP for large layers, FSDP for memory efficiency
- **TP + PP**: TP within stages, pipeline across stages

See `scripts/06_hybrid_parallelism/` for hybrid examples.

## References

- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053)
- [PyTorch Tensor Parallel](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
