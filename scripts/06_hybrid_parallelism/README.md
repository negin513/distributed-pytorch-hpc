# Hybrid Parallelism

Hybrid parallelism combines multiple parallelism strategies to train extremely large models efficiently. This is essential for models with billions to trillions of parameters.

## Why Hybrid Parallelism?

Single parallelism strategies have limitations:

- **DDP alone**: Model must fit on one GPU
- **FSDP alone**: Communication overhead increases with scale
- **TP alone**: Limited to number of GPUs per node, high communication cost

Combining strategies allows:
- Training models larger than any single GPU or node can handle
- Better scaling across hundreds to thousands of GPUs
- Optimal balance of memory efficiency and communication

## 2D Device Mesh — TP + FSDP

The standard approach combines TP within nodes (fast PCIe/NVLink) with
FSDP across nodes (Slingshot fabric). A 2D device mesh organizes the GPUs:

```
                        TP dimension (within node)
                    ┌─────────────────────────────────┐
                    │    GPU 0    GPU 1    GPU 2    GPU 3
               ┌────┤   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
               │    │   │TP=0 │ │TP=1 │ │TP=2 │ │TP=3 │
   FSDP        │Node│   │FSDP │ │FSDP │ │FSDP │ │FSDP │
   dimension   │ 0  │   │ =0  │ │ =0  │ │ =0  │ │ =0  │
   (across     │    │   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
   nodes)      ├────┤      │PCIe   │PCIe   │PCIe   │
               │    │ ─ ─ ─│─ ─ ─ ─│─ ─ ─ ─│─ ─ ─ ─│─ ─
               │    │   ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐
               │Node│   │TP=0 │ │TP=1 │ │TP=2 │ │TP=3 │
               │ 1  │   │FSDP │ │FSDP │ │FSDP │ │FSDP │
               │    │   │ =1  │ │ =1  │ │ =1  │ │ =1  │
               │    │   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
               ├────┤      │       │       │       │
               │    │ ─ ─Slingshot─│─ ─ ─ ─│─ ─ ─ ─│─ ─
               │    │   ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐
               │Node│   │TP=0 │ │TP=1 │ │TP=2 │ │TP=3 │
               │ 2  │   │FSDP │ │FSDP │ │FSDP │ │FSDP │
               │    │   │ =2  │ │ =2  │ │ =2  │ │ =2  │
               └────┤   └─────┘ └─────┘ └─────┘ └─────┘
                    └─────────────────────────────────┘

TP communication (all-reduce):  fast, within node (PCIe)
FSDP communication (all-gather/reduce-scatter): across nodes (Slingshot)
```

### Data Flow Through a Hybrid TP+FSDP Transformer Block

```
Each GPU holds: 1/4 of weights (TP) × 1/N_nodes of shards (FSDP)

  ┌─ FSDP all-gather weights for this layer ─┐
  │                                            │
  ▼                                            │
Input (each GPU has full batch chunk)          │
  │                                            │
  ├─ QKV projection (ColwiseParallel) ─┐      │
  │  Each GPU: X × W_qkv[:, chunk]     │      │
  │                                     │      │
  ├─ Attention (local per GPU) ─────────┤      │
  │                                     │      │
  ├─ Output projection (RowwiseParallel)│      │
  │  all-reduce across TP group ────────┘      │
  │                                            │
  ├─ LayerNorm (local) ───────────────────     │
  │                                            │
  ├─ FFN up (ColwiseParallel) ─────────┐      │
  │                                     │      │
  ├─ FFN down (RowwiseParallel)        │      │
  │  all-reduce across TP group ────────┘      │
  │                                            │
  ├─ LayerNorm (local) ───────────────────     │
  │                                            │
  └─ FSDP reduce-scatter gradients ────────────┘
```

## Common Hybrid Strategies

### 1. FSDP + TP (Recommended for Most Large Models)

**How it works:**
- TP within a node (fast PCIe on Derecho)
- FSDP across nodes (Slingshot fabric)

**Best for:**
- Large language models (10B+ parameters)
- Multi-node training on Derecho

### 2. DP + TP

**How it works:**
- TP within a node for large layers
- Data parallelism across nodes

**Best for:**
- Models with some very large layers
- Simpler than FSDP for certain architectures

### 3. 3D Parallelism (DP/FSDP + TP + PP)

**How it works:**
- Pipeline parallelism across nodes
- Tensor parallelism within pipeline stages
- Data/FSDP parallelism for each stage

**Best for:**
- Extremely large models (100B+ parameters)
- Maximum scalability (1000+ GPUs)

## Files in this Directory

### `01_fsdp_tp_hybrid.py`
Complete example combining FSDP and Tensor Parallelism.

**Key features:**
- 2D device mesh setup (FSDP x TP)
- Sharding strategies for hybrid parallelism
- LLaMA-style model training
- Proper initialization and synchronization

### `llama2_model.py`
LLaMA 2 model architecture optimized for hybrid parallelism.

**Key features:**
- Transformer architecture with TP support
- Column and row parallel linear layers
- Optimized attention mechanism
- RMSNorm implementation

### `log_utils.py`
Logging utilities for distributed training.

**Key features:**
- Rank-aware logging
- Memory usage tracking
- Performance metrics collection

## Running the Examples

### FSDP + TP on Multiple Nodes

```bash
# On a cluster with 4 nodes, 8 GPUs per node
# TP degree: 8 (within node)
# FSDP degree: 4 (across nodes)

# With mpiexec (recommended on Derecho)
mpiexec -n 32 --ppn 8 --cpu-bind none \
    python 01_fsdp_tp_hybrid.py \
    --tp_size=8 \
    --fsdp_size=4

# With torchrun
torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    01_fsdp_tp_hybrid.py \
    --tp_size=8 \
    --fsdp_size=4
```

## Key Concepts

### Device Mesh

A 2D mesh organizes GPUs for hybrid parallelism:

```python
from torch.distributed.device_mesh import init_device_mesh

# Create 2D mesh: 4 FSDP groups × 8 TP groups
mesh_2d = init_device_mesh(
    "cuda",
    (4, 8),
    mesh_dim_names=("fsdp", "tp")
)

fsdp_mesh = mesh_2d["fsdp"]  # For FSDP
tp_mesh = mesh_2d["tp"]      # For TP
```

### Applying Hybrid Parallelism

```python
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# 1. First apply TP
model = parallelize_module(
    model,
    tp_mesh,
    parallelize_plan
)

# 2. Then wrap with FSDP
model = FSDP(
    model,
    device_mesh=fsdp_mesh,
    sharding_strategy=ShardingStrategy.FULL_SHARD
)
```

### Order Matters!

The order of applying parallelism strategies is important:
1. **Inner-most first**: Apply TP first (within node)
2. **Outer-most last**: Apply FSDP last (across nodes)

## Performance Tips

1. **Match topology to hardware**
   - Use TP within nodes (fast NVLink)
   - Use FSDP/DP across nodes (slower interconnect)

2. **Balance parallelism degrees**
   - TP degree = GPUs per node (typically 8)
   - FSDP degree = Total GPUs / TP degree
   - Example: 32 GPUs = 4 FSDP × 8 TP

3. **Communication optimization**
   - Minimize cross-node communication
   - Overlap communication with computation
   - Use gradient accumulation to increase batch size

4. **Memory management**
   - Use mixed precision (BFloat16)
   - Enable gradient checkpointing for very large models
   - Consider CPU offloading for extreme cases

5. **Batch size scaling**
   - Global batch size = micro_batch × DP_degree × grad_accum_steps
   - Increase micro-batch size as much as memory allows

## Example Configurations (Derecho: 4 GPUs/node)

### Medium Model (~7B parameters)
- Hardware: 2 nodes × 4 GPUs = 8 GPUs
- Strategy: TP=4, FSDP=2
- Rationale: TP within each node, FSDP across 2 nodes

### Large Model (~13B parameters)
- Hardware: 4 nodes × 4 GPUs = 16 GPUs
- Strategy: TP=4, FSDP=4
- Rationale: Maximize FSDP for memory efficiency

### Very Large Model (~70B parameters)
- Hardware: 16 nodes × 4 GPUs = 64 GPUs
- Strategy: TP=4, FSDP=16
- Mixed precision: BFloat16
- Gradient checkpointing: Enabled

## Common Issues

### Out of Memory with Hybrid Parallelism
- Increase FSDP degree (more sharding)
- Reduce micro-batch size
- Enable activation checkpointing
- Use CPU offloading

### Poor Scaling Efficiency
- Check communication bottlenecks
- Ensure TP uses NVLink (within node)
- Increase batch size / gradient accumulation
- Profile to identify bottlenecks

### Incorrect Device Mesh Configuration
- Ensure total GPUs = FSDP_degree × TP_degree
- Verify mesh dimensions match your topology
- Check that mesh is created on all ranks

### Checkpoint Saving/Loading Issues
- Use FSDP state dict utilities
- Save with same mesh configuration
- Consider consolidating to rank 0 for easier loading

## Monitoring and Debugging

### Memory Usage
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Communication Profiling
```python
# Use PyTorch profiler
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Training step
    pass
```

### Throughput Metrics
Track:
- Samples per second
- Tokens per second (for LLMs)
- GPU utilization
- Communication time vs compute time

## References

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - NVIDIA's implementation
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [PyTorch Tensor Parallel](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [GPT-3 Training Paper](https://arxiv.org/abs/2005.14165)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
