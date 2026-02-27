#!/bin/bash
## Pipeline Parallelism Examples on Derecho
##
## Runs all pipeline parallelism examples.
## Uses launch_distributed() from pbs_common.sh (mpiexec launcher).
##
## To run manually without PBS:
##   mpiexec -n 4 --ppn 4 --cpu-bind none python 01_manual_model_split.py

#PBS -A SCSG0001
#PBS -N pipeline_parallel
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=64:ngpus=4:mem=480GB
#PBS -q main
#PBS -j oe

# Load shared PBS setup (modules, conda, NCCL config, node discovery)
source "$(dirname "${BASH_SOURCE[0]}")/../pbs_common.sh"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo ""
echo "============================================================"
echo "Pipeline Parallelism Examples"
echo "============================================================"
echo ""

# ----------------------------------------------------------
# Example 01: Manual model split with send/recv
# ----------------------------------------------------------
echo "--- Example 01: Manual model split (4 GPUs) ---"
launch_distributed "${SCRIPT_DIR}/01_manual_model_split.py"
echo ""

# ----------------------------------------------------------
# Example 02: Pipeline schedules (GPipe vs 1F1B)
# ----------------------------------------------------------
echo "--- Example 02: Pipeline schedules comparison (4 GPUs) ---"
launch_distributed "${SCRIPT_DIR}/02_pipeline_schedules.py"
echo ""

# ----------------------------------------------------------
# Example 03: Full pipeline training
# ----------------------------------------------------------
echo "--- Example 03: Pipeline training - 1F1B (4 GPUs) ---"
launch_distributed "${SCRIPT_DIR}/03_pipeline_training.py" \
    --schedule 1f1b --num-steps 20 --num-microbatches 4
echo ""

echo "--- Example 03: Pipeline training - GPipe (4 GPUs) ---"
launch_distributed "${SCRIPT_DIR}/03_pipeline_training.py" \
    --schedule gpipe --num-steps 20 --num-microbatches 4
echo ""

echo "============================================================"
echo "All pipeline parallelism examples completed."
echo "============================================================"
