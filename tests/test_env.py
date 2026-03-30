import os
import torch
import torch.distributed as dist

def format_nccl_version(version_tuple):
    if not version_tuple:
        return "N/A"
    return ".".join(map(str, version_tuple))

print("="*70)
print(" PyTorch / CUDA / NCCL Environment Test")
print("="*70)

# --- PyTorch version ---------------------------------------------------------
print(f"PyTorch version: {torch.__version__}")

# --- CUDA availability --------------------------------------------------------
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA runtime version (from torch): {torch.version.cuda}")

# --- cuDNN -------------------------------------------------------------------
print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
print(f"cuDNN version : {torch.backends.cudnn.version()}")

# --- NCCL --------------------------------------------------------------------
# torch.cuda.nccl.version() returns the runtime version of NCCL used by PyTorch
nccl_version = torch.cuda.nccl.version() if (torch.cuda.is_available() and hasattr(torch.cuda, 'nccl')) else None
print(f"NCCL available: {dist.is_nccl_available() if dist.is_available() else False}")
print(f"NCCL version  : {format_nccl_version(nccl_version)}")

# --- GPUs --------------------------------------------------------------------
if torch.cuda.is_available():
    print("\nDetected GPUs:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        print(f"       - SMs: {props.multi_processor_count}")
        print(f"       - Total Memory: {props.total_memory/1024**3:.2f} GB")
        print(f"       - Compute Capability: {props.major}.{props.minor}")
else:
    print("\nNo GPUs detected.")

# --- Current device ----------------------------------------------------------
if torch.cuda.is_available():
    dev = torch.cuda.current_device()
    print(f"\nCurrent CUDA device: {dev} -> {torch.cuda.get_device_name(dev)}")

# --- CUDA + LD_LIBRARY_PATH --------------------------------------------------
print("\nEnvironment Variables:")
print(f"  LD_LIBRARY_PATH = {os.environ.get('LD_LIBRARY_PATH', '(not set)')}")
print(f"  CUDA_HOME       = {os.environ.get('CUDA_HOME', '(not set)')}")

# --- NCCL smoke test ---------------------------------------------------------
if torch.cuda.is_available() and dist.is_nccl_available():
    print("\nRunning NCCL allreduce test (single-process)...")
    try:
        # 1. Setup minimal distributed environment for a single process
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        
        # 2. Initialize Process Group
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        
        # 3. Create tensor on GPU
        device = torch.device("cuda:0")
        tensor = torch.ones(2, device=device) * 2
        
        # 4. Run AllReduce
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        # 5. Cleanup
        dist.destroy_process_group()
        
        print("  SUCCESS: NCCL allreduce ran and process group destroyed.")
        print(f"  (Tensor value check: {tensor[0].item()})") # Should match input since world_size=1
        
    except Exception as e:
        print(f"  FAILURE: NCCL allreduce test failed.")
        print(f"  Error: {e}")
else:
    print("\nSkipping NCCL allreduce test (missing GPU or NCCL support).")

print("\nEnvironment test complete.")
print("="*70)
