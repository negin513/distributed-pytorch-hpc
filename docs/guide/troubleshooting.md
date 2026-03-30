# Troubleshooting Distributed PyTorch on Derecho

Common errors and their solutions.

## NCCL Errors

### NCCL Timeout / Hang at Startup

**Error:**
```
RuntimeError: NCCL communicator was aborted ... NCCL_TIMEOUT
```
or the job hangs indefinitely at `dist.init_process_group()`.

**Cause:** NCCL cannot establish communication between GPUs. Usually a
network interface issue on Derecho.

**Solution:**
```bash
# Always set on Derecho — this is the #1 fix
export NCCL_SOCKET_IFNAME=hsn
export NCCL_IB_DISABLE=1

# If still hanging, increase timeout
export NCCL_TIMEOUT=600000  # 10 minutes (default is 5 min)

# Debug: see what NCCL is doing
export NCCL_DEBUG=INFO
```

### NCCL Error: unhandled system error

**Error:**
```
NCCL error: unhandled system error, NCCL version 2.x.x
```

**Cause:** Slingshot network issue or misconfigured NCCL.

**Solution:**
```bash
export NCCL_SHM_DISABLE=1
export NCCL_CROSS_NIC=1
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_OPTIMIZED_MRS=false
```

## CUDA Errors

### CUDA Out of Memory (OOM)

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate X MiB (GPU 0; 40.00 GiB total capacity; ...)
```

**Cause:** Model + activations + optimizer states exceed GPU memory (40 GB on A100).

**Solutions:**
1. **Reduce batch size** — simplest fix
2. **Use FSDP** — shards model across GPUs
3. **Enable mixed precision** — halves activation memory
   ```python
   model = FSDP(model, mixed_precision=MixedPrecision(
       param_dtype=torch.bfloat16,
       reduce_dtype=torch.bfloat16,
   ))
   ```
4. **Gradient accumulation** — simulate larger batch with less memory
   ```python
   for micro_step in range(accum_steps):
       loss = model(data) / accum_steps
       loss.backward()
   optimizer.step()
   ```
5. **Activation checkpointing** — recompute instead of storing activations
   ```python
   from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
       apply_activation_checkpointing,
   )
   apply_activation_checkpointing(model)
   ```

### CUDA Error: device-side assert triggered

**Error:**
```
RuntimeError: CUDA error: device-side assert triggered
```

**Cause:** Usually an out-of-bounds index (e.g., label >= num_classes,
token ID >= vocab_size).

**Solution:**
```bash
# Run with CUDA_LAUNCH_BLOCKING to get a useful stack trace
CUDA_LAUNCH_BLOCKING=1 python train.py
```
Check that:
- Labels are in range `[0, num_classes)`
- Token IDs are in range `[0, vocab_size)`
- Tensor dimensions match what the model expects

## Distributed / Rank Errors

### Rank Mismatch / Wrong World Size

**Error:**
```
RuntimeError: ... expected world_size=8 but got 4
```

**Cause:** Mismatch between the number of processes launched and what the
script expects.

**Solution:**
Check your launch command:
```bash
# This launches 4 processes (single node):
mpiexec -n 4 --ppn 4 --cpu-bind none python train.py
torchrun --standalone --nproc_per_node=4 train.py

# This launches 8 processes (2 nodes x 4 GPUs):
mpiexec -n 8 --ppn 4 --cpu-bind none python train.py
```

Verify in the script:
```python
# Don't hardcode world_size — detect it
rank, world_size, local_rank = init_distributed()
```

### "Can't find environment variables for local rank"

**Error:**
```
SystemExit: Can't find the evironment variables for local rank
```

**Cause:** Script uses only one method for rank detection but was launched
with a different launcher.

**Solution:** Use `utils/distributed.py` which handles all launchers:
```python
from utils.distributed import init_distributed
rank, world_size, local_rank = init_distributed()
```

### Address Already in Use

**Error:**
```
RuntimeError: Address already in use
```

**Cause:** A previous job is still using the MASTER_PORT.

**Solution:**
```bash
# Use a different port
export MASTER_PORT=29501

# Or kill lingering processes
pkill -f python
```

## PBS / Job Scheduling Errors

### Job Stuck in Queue

**Cause:** Not enough GPU nodes available.

**Solution:**
```bash
# Check queue status
qstat -Q main

# Check your allocation
sacctmgr show associations user=$USER

# Try a shorter walltime (easier to schedule)
#PBS -l walltime=00:30:00
```

### "module: command not found" in PBS Script

**Cause:** PBS doesn't source your shell profile by default.

**Solution:** Add at the top of your PBS script:
```bash
source /etc/profile.d/modules.sh  # or wherever modules.sh lives
module purge
module load nvhpc cuda cray-mpich conda
```
Every PBS script in `scripts/` already includes these module loads — copy
any one as a starting template.

### Wrong Conda Environment

**Cause:** PBS jobs don't inherit your shell's conda activation.

**Solution:**
```bash
module load conda
conda activate ${CONDA_ENV:-pytorch-derecho}
# Verify:
which python
python -c "import torch; print(torch.__version__)"
```

## Performance Issues

### Slow Training (Low GPU Utilization)

**Diagnosis:**
```python
# Add to your training loop:
print(f"GPU util: {torch.cuda.utilization()}%")
print(f"Memory: {torch.cuda.memory_allocated()/1e9:.1f} GB / "
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
```

**Common causes:**
1. **Data loading bottleneck** — increase `num_workers` in DataLoader
2. **Small batch size** — GPU is underutilized
3. **Excessive communication** — reduce TP degree or use hybrid parallelism
4. **CPU-GPU transfer** — use `pin_memory=True` in DataLoader

### Communication Bottleneck

**Diagnosis:**
```python
# Profile communication vs compute
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    train_step()
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Solutions:**
1. Use TP within nodes, FSDP across nodes (minimize cross-node communication)
2. Increase batch size to amortize communication overhead
3. Use gradient accumulation
4. Enable NCCL OFI plugin for Slingshot
