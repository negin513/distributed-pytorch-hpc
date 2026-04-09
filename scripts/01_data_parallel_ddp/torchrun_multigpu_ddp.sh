#!/bin/bash
## ═══════════════════════════════════════════════════════════════════════
## DDP Multi-GPU Training on Derecho
##
## Launches distributed data-parallel examples with mpiexec.
## Copy this script and adjust #PBS and the python command for your job.
##
## Usage:
##   qsub torchrun_multigpu_ddp.sh -A <account>
## ═══════════════════════════════════════════════════════════════════════

#PBS -A SCSG0001
#PBS -N ddp_multigpu
#PBS -l walltime=01:00:00
#PBS -l select=2:ncpus=64:ngpus=4:mem=480GB
#PBS -q main
#PBS -j oe

# Load modules
module reset 
module load cuda conda mkl
conda activate pytorch-derecho
#conda activate torch28-nccl221-clone

#########################################
# Determine the number of nodes:
if [[ -z "${PBS_NODEFILE}" ]]; then
    echo "PBS_NODEFILE is not set. Assuming single-node job."
    nnodes=1
else
    nnodes=$(< $PBS_NODEFILE wc -l)
fi

if (( nnodes > 1)); then
    # Get a list of allocated nodes
    nodes=( $( cat $PBS_NODEFILE ) )
    echo nodes: $nodes

    ## --find headnode's IP:
    head_node=${nodes[0]}
    head_node_ip=$(ssh $head_node hostname -i | awk '{print $1}')

    echo "head_node_ip: $head_node_ip"
    
    # write the head node IP to a file 
    echo $(ssh $head_node hostname -i) > out
fi

GPUS_PER_NODE=4
TOTAL_PROCS=$((nnodes * GPUS_PER_NODE))

echo "═══════════════════════════════════════════════════"
echo "  Nodes:       $nnodes"
echo "  GPUs/node:   $GPUS_PER_NODE"
echo "  Total GPUs:  $TOTAL_PROCS"
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


echo "--- DDP basic training ---"
CMD="mpiexec -n $TOTAL_PROCS --cpu-bind none python multinode_ddp_basic.py --total_epochs 10"
echo "Running command: $CMD"
eval $CMD
