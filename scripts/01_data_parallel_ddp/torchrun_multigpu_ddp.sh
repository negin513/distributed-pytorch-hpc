#!/bin/bash
## DDP multi-GPU training on Derecho (mpiexec launcher)
##
## Usage:
##   qsub torchrun_multigpu_ddp.sh

#PBS -A SCSG0001
#PBS -N ddp_multigpu
#PBS -l walltime=01:00:00
#PBS -l select=2:ncpus=64:ngpus=4:mem=480GB
#PBS -q main
#PBS -j oe

# Load shared PBS setup (modules, conda, NCCL config, node discovery)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../pbs_common.sh"

# Run DDP examples
echo "--- DDP basic training ---"
launch_distributed "${SCRIPT_DIR}/multinode_ddp_basic.py" --total_epochs 10

echo ""
echo "--- DDP distributed dataloader ---"
launch_distributed "${SCRIPT_DIR}/distributed_dataloader.py"
