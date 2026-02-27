#!/bin/bash
## Tensor Parallelism examples on Derecho
##
## Usage:
##   qsub run_tensor_parallel.sh

#PBS -A SCSG0001
#PBS -N tensor_parallel
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=64:ngpus=4:mem=480GB
#PBS -q main
#PBS -j oe

# Load shared PBS setup (modules, conda, NCCL config, node discovery)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../pbs_common.sh"

echo "--- Example 01: Basic tensor parallel ---"
launch_distributed "${SCRIPT_DIR}/01_basic_tensor_parallel.py"
echo ""

echo "--- Example 02: Device mesh ---"
launch_distributed "${SCRIPT_DIR}/02_device_mesh_example.py"
echo ""

echo "--- Example 04: Advanced TP (1D mesh) ---"
launch_distributed "${SCRIPT_DIR}/04_advanced_tp_example.py"
echo ""

echo "All tensor parallelism examples completed."
