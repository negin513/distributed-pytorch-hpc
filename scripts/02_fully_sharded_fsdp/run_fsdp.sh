#!/bin/bash
## FSDP training on Derecho
##
## Usage:
##   qsub run_fsdp.sh

#PBS -A SCSG0001
#PBS -N fsdp_training
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=64:ngpus=4:mem=480GB
#PBS -q main
#PBS -j oe

# Load shared PBS setup (modules, conda, NCCL config, node discovery)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../pbs_common.sh"

TOTAL_PROCS=$((NNODES * GPUS_PER_NODE))

echo "--- FSDP ResNet-18 training (float32) ---"
mpiexec -n $TOTAL_PROCS --ppn $GPUS_PER_NODE --cpu-bind none \
    python "${SCRIPT_DIR}/resnet_fsdp_training.py" --epochs 10

echo ""
echo "--- FSDP ResNet-18 training (BFloat16 mixed precision) ---"
mpiexec -n $TOTAL_PROCS --ppn $GPUS_PER_NODE --cpu-bind none \
    python "${SCRIPT_DIR}/resnet_fsdp_training.py" --epochs 10 --use-amp
