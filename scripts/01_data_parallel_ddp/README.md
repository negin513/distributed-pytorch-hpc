# Distributed Data Parallel (DDP)

Distributed Data Parallel is PyTorch's recommended approach for multi-GPU training when the model fits on a single GPU. Each process/GPU maintains a complete copy of the model and processes different data batches.

![DDP Architecture](https://miro.medium.com/v2/resize:fit:720/format:webp/1*zbgdxY4VN_uola6kQWzlzw.png)

## How It Works

1. **Model Replication**: Each GPU gets a full copy of the model
2. **Data Distribution**: Training data is split across GPUs
3. **Forward Pass**: Each GPU processes its batch independently
4. **Gradient Sync**: Gradients are averaged across all GPUs using all-reduce
5. **Parameter Update**: All GPUs update their parameters (now identical)

## Files in this Directory

### `multinode_ddp_training.py`
Example script demonstrating DDP training across multiple nodes and GPUs.

### `torchrun_multigpu_ddp.sh`
PBS job script for running DDP training on Derecho.

**Usage:**
```bash
qsub torchrun_multigpu_ddp.sh
```

## Running the Examples

### Single Node, Multiple GPUs

```bash
torchrun --nproc_per_node=4 resnet_ddp_training.py
```

### Multiple Nodes (via PBS)

Edit `torchrun_multigpu_ddp.sh` to set:
- Number of nodes
- GPUs per node
- Training script and arguments

Then submit:
```bash
qsub torchrun_multigpu_ddp.sh
```

## Key Concepts

### Process Groups
DDP uses process groups for communication. Initialize with:
```python
import torch.distributed as dist
dist.init_process_group(backend='nccl')
```

### Wrapping Your Model
Wrap your model with DDP:
```python
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[local_rank])
```

### Data Loading
Use `DistributedSampler` to partition data:
```python
from torch.utils.data.distributed import DistributedSampler
sampler = DistributedSampler(dataset)
loader = DataLoader(dataset, sampler=sampler)
```

## Performance Tips

1. **Use NCCL backend** for GPU training (fastest for NVIDIA GPUs)
2. **Pin memory** in DataLoader: `pin_memory=True`
3. **Increase batch size** - scale with number of GPUs
4. **Adjust learning rate** - typically scale with batch size
5. **Enable gradient bucketing** - automatic in DDP for efficiency

## When to Use DDP

✅ **Good for:**
- Models that fit on a single GPU
- Scaling training across multiple GPUs/nodes
- Most standard training workloads

❌ **Not ideal for:**
- Models too large for a single GPU (use FSDP instead)
- Very large models where gradient communication is a bottleneck

## Common Issues

### Out of Memory
- Reduce batch size per GPU
- Consider gradient accumulation
- Try FSDP for memory-efficient training

### Gradient Synchronization Errors
- Ensure all processes execute the same number of backward passes
- Check that model architecture is identical across ranks

## References

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
