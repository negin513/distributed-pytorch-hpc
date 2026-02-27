#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# pbs_common.sh — Shared PBS setup for distributed PyTorch on Derecho
#
# Source from any PBS script:
#     source "$(dirname "${BASH_SOURCE[0]}")/pbs_common.sh"
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# -------------------------------------------------------------------
# 1. Module loading
# -------------------------------------------------------------------
module purge
module load nvhpc cuda cray-mpich conda

# -------------------------------------------------------------------
# 2. Conda environment
# -------------------------------------------------------------------
conda activate ${CONDA_ENV:-pytorch-derecho}

# -------------------------------------------------------------------
# 3. NCCL configuration for Derecho Slingshot
# -------------------------------------------------------------------
# Required: use the high-speed Slingshot network interfaces
export NCCL_SOCKET_IFNAME=hsn

# Disable InfiniBand (Derecho uses Slingshot/OFI, not IB)
export NCCL_IB_DISABLE=1

# Disable shared memory transport (avoids issues on some configs)
export NCCL_SHM_DISABLE=1

# Enable cross-NIC for non-rail-optimized networks
export NCCL_CROSS_NIC=1

# Number of channels per network peer
export NCCL_NCHANNELS_PER_NET_PEER=4

# Libfabric CXI settings (prevents CUDA deadlocks with Slingshot)
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd

# MPICH GPU settings
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1

# Restrict to available GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Debug level (set to INFO for debugging, VERSION for production)
export NCCL_DEBUG=${NCCL_DEBUG:-VERSION}

# -------------------------------------------------------------------
# 4. Node discovery from PBS
# -------------------------------------------------------------------
if [[ -z "${PBS_NODEFILE:-}" ]]; then
    NNODES=1
    HEAD_NODE_IP="127.0.0.1"
else
    NNODES=$(< "$PBS_NODEFILE" wc -l)
    NODES=( $(cat "$PBS_NODEFILE") )
    HEAD_NODE=${NODES[0]}

    if (( NNODES > 1 )); then
        HEAD_NODE_IP=$(ssh "$HEAD_NODE" hostname -i | awk '{print $1}')
    else
        HEAD_NODE_IP="127.0.0.1"
    fi
fi

GPUS_PER_NODE=${GPUS_PER_NODE:-4}

echo "═══════════════════════════════════════════════════"
echo "  Nodes:         $NNODES"
echo "  GPUs/node:     $GPUS_PER_NODE"
echo "  Total GPUs:    $((NNODES * GPUS_PER_NODE))"
echo "  Head node IP:  $HEAD_NODE_IP"
echo "  Conda env:     $CONDA_DEFAULT_ENV"
echo "═══════════════════════════════════════════════════"

# -------------------------------------------------------------------
# 5. launch_distributed() — run a script with mpiexec
# -------------------------------------------------------------------
# Usage:
#   launch_distributed script.py [script args...]
#
# Single-node:  mpiexec -n 4 --ppn 4 --cpu-bind none python script.py
# Multi-node:   mpiexec -n 8 --ppn 4 --cpu-bind none python script.py
#
# Rank detection is handled by utils/distributed.py which reads
# OMPI/PMI/mpi4py environment variables set by mpiexec.
#
launch_distributed() {
    local script="$1"
    shift

    local total_procs=$((NNODES * GPUS_PER_NODE))

    echo ">>> mpiexec -n $total_procs --ppn $GPUS_PER_NODE ($NNODES node(s) x $GPUS_PER_NODE GPUs)"
    mpiexec \
        -n "$total_procs" \
        --ppn "$GPUS_PER_NODE" \
        --cpu-bind none \
        python "$script" "$@"
}
