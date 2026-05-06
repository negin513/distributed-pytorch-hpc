#!/bin/bash
## ═══════════════════════════════════════════════════════════════════════
## Sequence Parallelism Examples on Derecho
##
## Launches sequence parallelism examples with mpiexec.
## Copy this script and adjust #PBS and the python command for your job.
##
## To run manually without PBS:
##   mpiexec -n 4 --ppn 4 --cpu-bind none python 01_basic_sequence_parallel.py
##
## Usage:
##   qsub run_sequence_parallel.sh
## ═══════════════════════════════════════════════════════════════════════

#PBS -A SCSG0001
#PBS -N sequence_parallel
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=64:ngpus=4:mem=480GB
#PBS -q main
#PBS -j oe

set -euo pipefail

# ── Modules & Conda ──────────────────────────────────────────────────
module purge
module load nvhpc cuda cray-mpich conda
conda activate ${CONDA_ENV:-pytorch-derecho}

# ── NCCL Configuration for Derecho Slingshot ─────────────────────────
export NCCL_SOCKET_IFNAME=hsn          # Use Slingshot high-speed network
export NCCL_IB_DISABLE=1               # Derecho uses OFI, not InfiniBand
export NCCL_SHM_DISABLE=1              # Avoid shared-memory transport issues
export NCCL_CROSS_NIC=1                # Enable cross-NIC communication
export NCCL_NCHANNELS_PER_NET_PEER=4   # Channels per network peer

# Libfabric CXI settings (prevents CUDA deadlocks on Slingshot)
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd

# MPICH GPU-aware MPI
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=${NCCL_DEBUG:-VERSION}

# ── Node Discovery ───────────────────────────────────────────────────
if [[ -z "${PBS_NODEFILE:-}" ]]; then
    NNODES=1
else
    NNODES=$(< "$PBS_NODEFILE" wc -l)
fi
GPUS_PER_NODE=4
TOTAL_PROCS=$((NNODES * GPUS_PER_NODE))

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "═══════════════════════════════════════════════════"
echo "  Nodes:       $NNODES"
echo "  GPUs/node:   $GPUS_PER_NODE"
echo "  Total GPUs:  $TOTAL_PROCS"
echo "═══════════════════════════════════════════════════"

# ── Run Examples ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Sequence Parallelism Examples"
echo "============================================================"
echo ""

echo "--- Example 01: Basic SP concepts (4 GPUs) ---"
mpiexec -n $TOTAL_PROCS --ppn $GPUS_PER_NODE --cpu-bind none \
    python "${SCRIPT_DIR}/01_basic_sequence_parallel.py"
echo ""

echo "--- Example 02: SP transformer block with PyTorch API (4 GPUs) ---"
mpiexec -n $TOTAL_PROCS --ppn $GPUS_PER_NODE --cpu-bind none \
    python "${SCRIPT_DIR}/02_sp_transformer_layer.py"
echo ""

echo "--- Example 03: SP+TP training (seq_len=512, 4 GPUs) ---"
mpiexec -n $TOTAL_PROCS --ppn $GPUS_PER_NODE --cpu-bind none \
    python "${SCRIPT_DIR}/03_sp_training.py" \
    --seq-len 512 --num-steps 20
echo ""

echo "--- Example 03: SP+TP training (seq_len=2048, 4 GPUs) ---"
mpiexec -n $TOTAL_PROCS --ppn $GPUS_PER_NODE --cpu-bind none \
    python "${SCRIPT_DIR}/03_sp_training.py" \
    --seq-len 2048 --num-steps 10
echo ""

echo "--- Example 04: DeepSpeed-Ulysses SP (4 GPUs) ---"
mpiexec -n $TOTAL_PROCS --ppn $GPUS_PER_NODE --cpu-bind none \
    python "${SCRIPT_DIR}/04_ulysses_sequence_parallel.py"
echo ""

echo "--- Example 05: Ring Attention concept (4 GPUs) ---"
mpiexec -n $TOTAL_PROCS --ppn $GPUS_PER_NODE --cpu-bind none \
    python "${SCRIPT_DIR}/05_ring_attention_concept.py"
echo ""

echo "============================================================"
echo "All sequence parallelism examples completed."
echo "============================================================"
