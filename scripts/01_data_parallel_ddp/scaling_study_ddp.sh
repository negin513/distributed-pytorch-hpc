#!/bin/bash
## ═══════════════════════════════════════════════════════════════════════
## DDP U-Net Scaling Study on Derecho
##
## Runs multinode_ddp_unet.py across multiple (nodes × gpus/node) layouts
## to produce a weak/strong scaling table. Allocates 2 nodes × 4 GPUs and
## sweeps: 1x1, 1x2, 2x1, 1x4, 2x2, 2x3, 2x4.
##
## Usage:
##   qsub scaling_study_ddp.sh -A <account>
## ═══════════════════════════════════════════════════════════════════════

#PBS -A SCSG0001
#PBS -N ddp_scaling
#PBS -l walltime=01:00:00
#PBS -l select=2:ncpus=64:ngpus=4:mem=480GB
#PBS -q main
#PBS -j oe

# Load modules
module reset
module load cuda conda mkl
conda activate pytorch-derecho

#########################################
# Determine the number of nodes:
if [[ -z "${PBS_NODEFILE}" ]]; then
    echo "PBS_NODEFILE is not set. Assuming single-node job."
    nnodes=1
else
    nnodes=$(< $PBS_NODEFILE wc -l | awk '{print $1}')
    # PBS_NODEFILE lists one line per requested slot; dedupe to nodes.
    nnodes=$(sort -u $PBS_NODEFILE | wc -l)
fi

if (( nnodes > 1 )); then
    nodes=( $( sort -u $PBS_NODEFILE ) )
    head_node=${nodes[0]}
    head_node_ip=$(ssh $head_node hostname -i | awk '{print $1}')
    echo "head_node_ip: $head_node_ip"
fi

GPUS_PER_NODE=4
TOTAL_GPUS=$((nnodes * GPUS_PER_NODE))

echo "═══════════════════════════════════════════════════"
echo "  Allocated nodes : $nnodes"
echo "  GPUs/node       : $GPUS_PER_NODE"
echo "  Total GPUs      : $TOTAL_GPUS"
echo "═══════════════════════════════════════════════════"
#########################################

# ── NCCL Configuration for Derecho Slingshot ─────────────────────────
export NCCL_SOCKET_IFNAME=hsn          # Use Slingshot high-speed network (NEEDED)
export NCCL_IB_DISABLE=1               # Derecho uses OFI, not InfiniBand
export NCCL_SHM_DISABLE=1              # Avoid shared-memory transport issues
export NCCL_CROSS_NIC=1                # Enable cross-NIC communication
export NCCL_NCHANNELS_PER_NET_PEER=4   # Channels per network peer

# Libfabric CXI settings (prevents CUDA deadlocks on Slingshot)
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_OPTIMIZED_MRS=false
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd

# MPICH GPU-aware MPI
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1
export LSCRATCH=/glade/derecho/scratch/$USER/

#########################################
# Training hyperparameters (same for all configs so results are comparable)
NUM_EPOCHS=4
NUM_SAMPLES=1600
BATCH_SIZE=16

# Output directory for per-config logs
LOGDIR=./scaling_logs_${PBS_JOBID:-local}
mkdir -p "$LOGDIR"
echo "Logs will be written to: $LOGDIR"
echo ""

# Sweep layouts: "label:nodes:ppn"
CONFIGS=(
    "1x1:1:1"
    "1x2:1:2"
    "2x1:2:1"
    "1x4:1:4"
    "2x2:2:2"
    "2x3:2:3"
    "2x4:2:4"
)

run_config() {
    local label=$1
    local nodes=$2
    local ppn=$3
    local total=$(( nodes * ppn ))
    local logfile="$LOGDIR/run_${label}.log"

    if (( nodes > nnodes )); then
        echo "[SKIP] $label needs $nodes nodes but only $nnodes allocated"
        return
    fi
    if (( ppn > GPUS_PER_NODE )); then
        echo "[SKIP] $label needs $ppn GPUs/node but only $GPUS_PER_NODE available"
        return
    fi

    echo "─────────────────────────────────────────────────"
    echo "  Config $label  |  nodes=$nodes  ppn=$ppn  total=$total GPUs"
    echo "─────────────────────────────────────────────────"
    CMD="mpiexec -n $total --ppn $ppn --cpu-bind none python multinode_ddp_unet.py \
         --num_epochs $NUM_EPOCHS --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE"
    echo "  $CMD"
    eval "$CMD" > "$logfile" 2>&1
    local rc=$?
    if (( rc != 0 )); then
        echo "  [FAIL] rc=$rc — see $logfile"
    else
        # Pull steady-state (epochs 2..N) average from the per-epoch lines.
        awk '
            /Epoch [0-9]+\/[0-9]+ .* Throughput: / {
                match($0, /Epoch ([0-9]+)\/([0-9]+)/, a)
                match($0, /Throughput: ([0-9.]+)/, b)
                if (a[1] >= 2) { sum += b[1]; n++ }
            }
            END {
                if (n > 0) printf "  Steady throughput (epochs 2..N): %.1f samples/s  (per-GPU: %.1f)\n", sum/n, sum/n/TOTAL
            }
        ' TOTAL=$total "$logfile"
    fi
    echo ""
}

for cfg in "${CONFIGS[@]}"; do
    IFS=":" read -r label nodes ppn <<< "$cfg"
    run_config "$label" "$nodes" "$ppn"
done

#########################################
# Final summary table
echo "═══════════════════════════════════════════════════"
echo "  Scaling Study Summary"
echo "  bs=$BATCH_SIZE  samples=$NUM_SAMPLES  epochs=$NUM_EPOCHS"
echo "═══════════════════════════════════════════════════"
printf "  %-6s %-7s %-7s %-18s %-12s\n" "Label" "Nodes" "GPUs" "Throughput(s/s)" "Per-GPU"
printf "  %-6s %-7s %-7s %-18s %-12s\n" "-----" "-----" "----" "---------------" "-------"
for cfg in "${CONFIGS[@]}"; do
    IFS=":" read -r label nodes ppn <<< "$cfg"
    total=$(( nodes * ppn ))
    logfile="$LOGDIR/run_${label}.log"
    if [[ ! -f "$logfile" ]]; then
        printf "  %-6s %-7s %-7s %-18s %-12s\n" "$label" "$nodes" "$total" "SKIPPED" "-"
        continue
    fi
    result=$(awk '
        /Epoch [0-9]+\/[0-9]+ .* Throughput: / {
            match($0, /Epoch ([0-9]+)\/[0-9]+/, a)
            match($0, /Throughput: ([0-9.]+)/, b)
            if (a[1] >= 2) { sum += b[1]; n++ }
        }
        END {
            if (n > 0) printf "%.1f %.1f", sum/n, sum/n/TOTAL
            else       printf "FAILED -"
        }
    ' TOTAL=$total "$logfile")
    tput=$(echo "$result" | awk '{print $1}')
    pgpu=$(echo "$result" | awk '{print $2}')
    printf "  %-6s %-7s %-7s %-18s %-12s\n" "$label" "$nodes" "$total" "$tput" "$pgpu"
done
echo "═══════════════════════════════════════════════════"
