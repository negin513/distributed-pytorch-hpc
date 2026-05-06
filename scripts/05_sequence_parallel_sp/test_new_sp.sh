#!/bin/bash
## Quick test of the two new SP scripts (04_ulysses, 05_ring)
#PBS -A SCSG0001
#PBS -N test_new_sp
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=64:ngpus=4:mem=480GB
#PBS -q main
#PBS -j oe

set -euo pipefail

module purge
module load nvhpc cuda cray-mpich conda
conda activate ${CONDA_ENV:-torch28-nccl221}

export NCCL_SOCKET_IFNAME=hsn
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_CROSS_NIC=1
export NCCL_NCHANNELS_PER_NET_PEER=4
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=VERSION

SCRIPT_DIR="/glade/work/negins/distributed-pytorch-hpc/scripts/05_sequence_parallel_sp"

echo "============================================================"
echo "Testing new SP scripts (4 GPUs, 1 node)"
echo "============================================================"
echo ""

echo "--- 04_ulysses_sequence_parallel.py ---"
mpiexec -n 4 --ppn 4 --cpu-bind none \
    python "${SCRIPT_DIR}/04_ulysses_sequence_parallel.py"
echo ""

echo "--- 05_ring_attention_concept.py ---"
mpiexec -n 4 --ppn 4 --cpu-bind none \
    python "${SCRIPT_DIR}/05_ring_attention_concept.py"
echo ""

echo "============================================================"
echo "All tests completed."
echo "============================================================"
