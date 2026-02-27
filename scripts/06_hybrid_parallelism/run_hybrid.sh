#!/bin/bash
## Hybrid FSDP + TP training on Derecho
##
## Usage:
##   qsub run_hybrid.sh

#PBS -A SCSG0001
#PBS -N hybrid_fsdp_tp
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=64:ngpus=4:mem=480GB
#PBS -q main
#PBS -j oe

# Load shared PBS setup (modules, conda, NCCL config, node discovery)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../pbs_common.sh"

echo "--- FSDP + TP hybrid (LLaMA-style model) ---"
launch_distributed "${SCRIPT_DIR}/01_fsdp_tp_hybrid.py"

echo ""
echo "Hybrid parallelism example completed."
