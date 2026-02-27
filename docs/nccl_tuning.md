# NCCL Tuning Guide for Derecho

NCCL (NVIDIA Collective Communication Library) handles all GPU-to-GPU
communication for distributed PyTorch training. Proper configuration
is critical on Derecho's Slingshot fabric.

## Required Settings for Derecho

These must be set for NCCL to work on Derecho:

```bash
# Use Slingshot high-speed network interfaces (hsn0, hsn1, ...)
# Without this, NCCL tries Ethernet and hangs or is extremely slow
export NCCL_SOCKET_IFNAME=hsn

# Disable InfiniBand — Derecho uses Slingshot/OFI, not IB
export NCCL_IB_DISABLE=1
```

## Recommended Settings

```bash
# Disable shared memory transport
# Avoids occasional hangs on some node configurations
export NCCL_SHM_DISABLE=1

# Enable cross-NIC communication
# Improves performance on non-rail-optimized networks like Slingshot
export NCCL_CROSS_NIC=1

# Number of channels per network peer
# More channels = more parallelism in communication
# 4 is a good default; increase for large messages
export NCCL_NCHANNELS_PER_NET_PEER=4
```

## MPICH GPU Settings

These enable GPU-aware MPI operations through Cray MPICH:

```bash
# Enable GPU memory support in MPICH
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1

# GPU-aware NIC selection policy
export MPICH_OFI_NIC_POLICY=GPU

# Enable GPU-direct support
export MPICH_GPU_SUPPORT_ENABLED=1
```

## Libfabric (CXI) Settings

These libfabric settings improve stability with CUDA and large-scale jobs
on Slingshot:

```bash
# Prevent CUDA allocation calls from causing deadlocks with libfabric
export FI_CXI_DISABLE_HOST_REGISTER=1

# Increase completion queue size for large jobs (default is too small)
export FI_CXI_DEFAULT_CQ_SIZE=131072

# Use userfaultfd for memory registration cache monitoring
export FI_MR_CACHE_MONITOR=userfaultfd
```

`FI_CXI_DISABLE_HOST_REGISTER=1` is particularly important — without it,
CUDA memory allocations can deadlock with the CXI provider's host memory
registration.

## Understanding NCCL Transport on Slingshot

NCCL can communicate over Derecho's Slingshot fabric in two ways:

### 1. Socket Transport (Default with generic PyTorch)

A generic PyTorch install (e.g. `conda install pytorch` from the default
channel) bundles a generic NCCL that uses **TCP sockets** over the
Slingshot interfaces (`hsn0`, `hsn1`). This is what `NCCL_SOCKET_IFNAME=hsn`
enables.

- **Functional** but not optimal for multi-node training
- Higher latency and lower bandwidth than native OFI transport
- No GPU Direct RDMA — data copies through host memory

> **Note:** This repo's `environment.yml` uses Derecho-optimized PyTorch
> builds with an NCCL built for Cray MPICH + Slingshot, which may already
> include OFI transport support.

### 2. Native OFI Transport (Requires NCCL + AWS OFI Plugin built from source)

The [AWS OFI NCCL Plugin](https://github.com/aws/aws-ofi-nccl) provides
a native libfabric transport for NCCL, using Slingshot's CXI provider
directly. This bypasses TCP sockets entirely.

- Significantly better multi-node bandwidth and latency
- Enables GPU Direct RDMA (GPU memory ↔ network, no host copy)
- Requires building NCCL and the OFI plugin from source against
  Derecho's system libfabric (`/opt/cray/libfabric`)
- **Recommended for production multi-node training**

### How to Get the OFI Transport

Building NCCL + the OFI plugin from source is nontrivial. Ben Kirk's
[derecho-pytorch-mpi](https://github.com/benkirk/derecho-pytorch-mpi)
repo provides a reproducible build process that:

1. Compiles NCCL from source (e.g., v2.21.5) targeting A100 (`sm_80`)
2. Builds the AWS OFI NCCL Plugin (e.g., v1.6.0) against Cray's
   libfabric (e.g., `/opt/cray/libfabric/1.15.2.0`)
3. Builds PyTorch from source linked to this custom NCCL
4. Installs a conda environment with activation scripts that set all
   required runtime variables

If you have access to a pre-built environment (check with your system
admins or `ls /glade/work/benkirk/conda-envs/`), you can activate it
directly.

### Runtime Settings for OFI Transport

When using a PyTorch build with the OFI plugin:

```bash
# Tell NCCL to use the OFI network (fails fast if plugin not available)
export NCCL_NET="AWS Libfabric"

# MPICH GPU settings
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU

# Libfabric CXI settings
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd

# Cross-NIC for multi-rail
export NCCL_CROSS_NIC=1

# Use Slingshot interfaces for bootstrap
export NCCL_SOCKET_IFNAME=hsn
```

### Verifying Which Transport NCCL Uses

```bash
export NCCL_DEBUG=INFO
# Then run your training script and look for:
#   "NCCL INFO Using network AWS Libfabric"   → OFI transport (good)
#   "NCCL INFO Using network Socket"          → TCP sockets (default)
```

### Performance Comparison

| Transport | Multi-Node Bandwidth | Latency | GPU Direct RDMA |
|-----------|---------------------|---------|-----------------|
| Socket (default) | ~10-15 GB/s | Higher | No |
| OFI + libfabric | ~20-25 GB/s | Lower | Yes |

> **Bottom line:** The examples in this repo work correctly with the
> default socket transport. For production multi-node training at scale,
> the OFI transport provides meaningfully better performance. Start with
> the defaults to get your code working, then upgrade the NCCL backend
> when scaling up.

## Debug Settings

### Minimal (Production)

```bash
export NCCL_DEBUG=VERSION    # Just print NCCL version at startup
```

### Standard Debugging

```bash
export NCCL_DEBUG=INFO       # Shows initialization, topology, algorithms
```

### Verbose Debugging

```bash
export NCCL_DEBUG=TRACE      # Every collective call (very verbose!)
export NCCL_DEBUG_FILE=/path/to/nccl_%h_%p.log  # Per-host log files
export NCCL_DEBUG_SUBSYS=ALL  # All subsystems
```

### PyTorch-Level Debugging

```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Log all dist.* calls
```

## Full Environment Variable Reference

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `NCCL_SOCKET_IFNAME` | Interface name(s) | auto | Network interface for socket connections |
| `NCCL_IB_DISABLE` | 0, 1 | 0 | Disable InfiniBand transport |
| `NCCL_SHM_DISABLE` | 0, 1 | 0 | Disable shared memory transport |
| `NCCL_CROSS_NIC` | 0, 1, 2 | 0 | Cross-NIC communication mode |
| `NCCL_P2P_DISABLE` | 0, 1 | 0 | Disable GPU peer-to-peer |
| `NCCL_P2P_LEVEL` | LOC, NVL, PIX, PXB, PHB, SYS | auto | P2P communication level |
| `NCCL_NET_GDR_LEVEL` | LOC, PIX, PXB, PHB, SYS | auto | GPU Direct RDMA level |
| `NCCL_NCHANNELS_PER_NET_PEER` | 1-32 | auto | Channels per network peer |
| `NCCL_MAX_NCHANNELS` | 1-32 | auto | Maximum channels per connection |
| `NCCL_MIN_NCHANNELS` | 1-32 | auto | Minimum channels per connection |
| `NCCL_BUFFSIZE` | bytes | 4194304 | Buffer size per channel |
| `NCCL_NTHREADS` | 64-1024 | auto | CUDA threads per NCCL kernel |
| `NCCL_ALGO` | Ring, Tree, CollNetDirect, CollNetChain | auto | Collective algorithm |
| `NCCL_PROTO` | Simple, LL, LL128 | auto | Communication protocol |
| `NCCL_DEBUG` | VERSION, WARN, INFO, TRACE | WARN | Debug log level |
| `NCCL_DEBUG_FILE` | path | stderr | Debug output file |
| `NCCL_DEBUG_SUBSYS` | INIT, NET, ... ALL | ALL | Debug subsystems |
| `NCCL_TIMEOUT` | ms | 300000 | Watchdog timeout (5 min default) |
| `NCCL_ASYNC_ERROR_HANDLING` | 0, 1 | 1 | Enable async error handling |

## Troubleshooting Common NCCL Issues

### Hang at init_process_group

```
Cause: NCCL can't find the right network interface
Fix:   export NCCL_SOCKET_IFNAME=hsn
Debug: export NCCL_DEBUG=INFO  (check which interface NCCL picks)
```

### "NCCL WARN ... no CUDA-capable device"

```
Cause: CUDA_VISIBLE_DEVICES not set or GPU not accessible
Fix:   export CUDA_VISIBLE_DEVICES=0,1,2,3
Debug: python -c "import torch; print(torch.cuda.device_count())"
```

### Slow Multi-Node Performance

```
Cause: NCCL using wrong transport (e.g., TCP instead of OFI)
Fix:   Ensure NCCL_SOCKET_IFNAME=hsn and consider OFI plugin
Debug: export NCCL_DEBUG=INFO  (look for "Using network" line)
```

### NCCL Timeout

```
Cause: One rank is stuck or deadlocked
Fix:   Check for mismatched collective calls across ranks
Debug: export NCCL_DEBUG=INFO
       export TORCH_DISTRIBUTED_DEBUG=DETAIL
```
