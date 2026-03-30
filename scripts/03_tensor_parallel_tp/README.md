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

### `01_basic_tensor_parallel.py`
Introduction to tensor parallelism basics.

**Key concepts:**
- Simple tensor splitting across GPUs
- Basic collective operations
- Understanding tensor distribution

### `02_device_mesh_example.py`
Using PyTorch's DeviceMesh for organizing GPUs.

**Key concepts:**
- Creating multi-dimensional device meshes
- Organizing GPUs for different parallelism strategies
- Foundation for hybrid parallelism

### `03_2d_tensor_parallel.py`
2D tensor parallelism (2D-TP) for large models.

**Key concepts:**
- Combining row and column parallelism
- Reducing communication overhead
- Scaling to larger GPU counts

### `04_advanced_tp_example.py`
Advanced tensor parallelism patterns.

**Key concepts:**
- Sharding strategies for different layer types
- Custom parallelization patterns
- Integration with transformer architectures

## Running the Examples

### Basic Example
```bash
mpiexec -n 4 --ppn 4 --cpu-bind none python 01_basic_tensor_parallel.py
torchrun --standalone --nproc_per_node=4 01_basic_tensor_parallel.py
```

### 2D Tensor Parallelism (8 GPUs)
```bash
mpiexec -n 8 --ppn 4 --cpu-bind none python 03_2d_tensor_parallel.py
torchrun --nproc_per_node=4 03_2d_tensor_parallel.py   # single node
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

# 2D mesh for TP + DP
mesh_2d = init_device_mesh("cuda", (2, 4))  # 2 TP groups of 4 GPUs each
```

### Communication Patterns

TP requires specific communication patterns:

- **All-Gather**: Collect tensor shards from all GPUs
- **Reduce-Scatter**: Reduce and distribute results
- **All-Reduce**: Sum gradients across all GPUs

## Performance Tips

1. **Minimize communication** - group operations when possible
2. **Use NCCL** - optimized for GPU-to-GPU communication
3. **Balance sharding** - ensure equal work distribution
4. **Overlap communication and computation** - hide communication latency
5. **Consider 2D TP** - reduces communication for large models

## When to Use Tensor Parallelism

✅ **Good for:**
- Very large layers that don't fit on a single GPU
- Transformer models with large hidden dimensions
- Models with large embedding tables
- When combined with other parallelism strategies (hybrid)

❌ **Not ideal for:**
- Small models where communication overhead is high
- When GPUs are connected via slow interconnect
- As the only parallelism strategy for medium-sized models

## Comparison with Other Strategies

| Strategy | Model Size | Communication | Memory Efficiency |
|----------|-----------|---------------|-------------------|
| DDP | Replicated | Gradients only | Low (full replica) |
| FSDP | Sharded | Parameters + gradients | High |
| TP | Distributed | Activations + gradients | Medium-High |

## 1D vs 2D Tensor Parallelism

### 1D TP
- Simpler implementation
- Higher communication volume
- Good for 2-8 GPUs

### 2D TP
- More complex setup
- Reduced communication (√N instead of N)
- Better for 8+ GPUs
- Requires 2D device mesh

## Common Issues

### High Communication Overhead
- Ensure GPUs are connected via NVLink or high-speed interconnect
- Consider reducing TP degree
- Use 2D TP to reduce communication
- Overlap communication with computation

### Incorrect Tensor Shapes
- Ensure layer dimensions are divisible by TP degree
- Adjust model architecture for TP compatibility
- Use padding if necessary

### Load Imbalance
- Ensure uniform distribution of work
- Profile to identify bottlenecks
- Consider hybrid parallelism strategies

## Combining with Other Strategies

TP is often combined with:

- **TP + DP**: Tensor parallel within nodes, data parallel across nodes
- **TP + FSDP**: TP for large layers, FSDP for memory efficiency
- **TP + PP**: TP within stages, pipeline across stages

See `examples/06_hybrid_parallelism/` for examples.

## References

- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053)
- [PyTorch Tensor Parallel](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [2D Tensor Parallelism](https://arxiv.org/abs/2104.04473)
