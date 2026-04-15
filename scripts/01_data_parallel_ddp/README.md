# Distributed Data Parallel (DDP)

Data Parallel is the most common distributed training strategy and usually the first one you should try when scaling up your training workflows. The idea is simple: every GPU has a complete copy of the full model, but each processes a different slice of the data, because data doesn't fit on one GPU or training is too slow. 

PyTorch DistributedDataParallel (DDP) is the standard implementation of this strategy.
DDP works by running one process per GPU, where:
- Each process has its own full copy of the model
- Each process computes forward and backward passes on its own batch (shard) of data
- Gradients are averaged across all processes using an efficient collective communication operation called `all-reduce` after each backward pass, ensuring that all model replicas remain in sync.

DDP is PyTorch's recommended approach and easiest to implement for multi-GPU training and works across both single-node and multi-node setups.

## How DDP Works?

1. **Model Replication**: Each GPU gets a full copy of the model
2. **Data Distribution**: Training data is split across GPUs using `DistributedSampler` to ensure each GPU processes a unique subset of the data
3. **Forward Pass**: Each GPU processes its batch independently
4. **Gradient Sync**: During backward propagation, gradients are averaged across all GPUs using `all-reduce` operation (typically via NCCL backend for GPUs)
5. **Parameter Update**: All GPUs update their parameters and after synchronization, each model replica remains identical.

![DDP Architecture](https://miro.medium.com/v2/resize:fit:720/format:webp/1*zbgdxY4VN_uola6kQWzlzw.png)

Each GPU computes gradients independently, then participates in an all-reduce operation to ensure all model replicas remain identical before the next optimization step.


## How to modify your training script for DDP?

There are only a few changes needed to convert a single-GPU training script to DDP. The main steps are:
1. Initialize the distributed process group
2. Set the device for each process using `local_rank`
3. Wrap your model with `DistributedDataParallel`
4. Use `DistributedSampler` for your dataset to ensure proper data sharding
5. Set the epoch on the sampler at the start of each epoch to ensure different shuffling each epoch.    
6. Keep the training loop the same (zero_grad → forward → backward → step)
7. Clean up the process group at the end of training.


### 1. Initialize the process group:
Each DDP process needs to initialize the process group for communication. This is typically done at the start of your script:
```python
import torch.distributed as dist
dist.init_process_group(backend="nccl")  # Use NCCL backend for GPU training
```

### 2. Set the device for each process:
Each process should set its device based on the `LOCAL_RANK` environment variable, which is automatically set by the launcher:
```python
import os
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
```

The launcher (e.g., `torchrun`) will set `LOCAL_RANK`, `WORLD_RANK` and `WORLD_SIZE` to a unique value for each process, allowing you to assign each process to a different GPU. For example, if you have 4 GPUs, the processes will have `LOCAL_RANK` values of 0, 1, 2, and 3, which correspond to the GPU IDs.

The utility function in `utils/distributed.py` can help with this setup and also provides a `setup_distributed()` function that initializes the process group and sets the device for you based on the environment variables with different launchers (e.g., `torchrun`, `mpiexec`, etc.). You can call this function at the start of your training script to handle the distributed setup.

### 3. Wrap your model with `DistributedDataParallel`:
The DDP wrapper automatically synchronizes gradients across all GPUs during the backward pass. Move the model to the device before wrapping.

```python
from torch.nn.parallel import DistributedDataParallel as DDP
model = model.to(LOCAL_RANK)
model = DDP(model, device_ids=[LOCAL_RANK])
```

### 4. Use `DistributedSampler` for your dataset:
To ensure that each GPU processes a unique subset of the data, use `DistributedSampler`:
```python
from torch.utils.data.distributed import DistributedSampler
train_loader = DataLoader(
        train_dataset,
        sampler=DistributedSampler(train_dataset), 
        batch_size=batch_size,
        shuffle=False,                          # shuffle is handled by the sampler
        pin_memory=True,
        ...)
```
Set `shuffle=False` in the DataLoader—the sampler handles shuffling by setting the epoch, i.e. `sampler.set_epoch(epoch)` at the start of each epoch.
Note that the effective batch size becomes `batch_size × num_gpus`.

### 5. Destroy the process group at the end of training:

Destroy the process group to clean up resources:
```python
dist.destroy_process_group()
```


## How to launch DDP training?
Use `torchrun` for single-node multi-GPU training, or `mpiexec` or `mpirun` for multi-node setups.

```
# Single-node multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 resnet_ddp_training.py

# Multi-node (2 nodes, 4 GPUs each)
mpiexec -n 8 --ppn 4 --cpu-bind none python multinode_ddp_unet.py
```

## Files in this Directory

| File | Description |
|------|-------------|
| `multinode_ddp_basic.py` | Minimal DDP template with a synthetic Linear-model dataset |
| `multinode_ddp_unet.py` | Full DDP U-Net example on synthetic ERA5-like data |
| `distributed_dataloader.py` | Focused `DistributedSampler` + `DataLoader` example |
| `torchrun_multigpu_ddp.sh` | PBS job script for a single DDP run on Derecho |

### Running via PBS

Edit `torchrun_multigpu_ddp.sh` to set nodes, GPUs, project key, and training arguments, then submit:
```bash
qsub torchrun_multigpu_ddp.sh
```


-----------------------------------------------------------
## 3 Key Concepts

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
4. **Adjust learning rate** - typically scale *linearly* with batch size
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

### Out of Memory (OOM)
- Reduce batch size per GPU
- Consider gradient accumulation
- Try FSDP for memory-efficient training

### Gradient Synchronization Errors
- Ensure all processes execute the same number of backward passes
- Check that model architecture is identical across ranks

### Hanging or Slow Training
- Verify `MASTER_ADDR` and `MASTER_PORT` are set correctly for multi-node
- Ensure all nodes can communicate with NCCL backend

## References

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
