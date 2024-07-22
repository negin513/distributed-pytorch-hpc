#!/bin/bash

## This script is for running a multi-GPU job on Derecho using torchrun and MPI. 


#PBS -A SCSG0001         
#PBS -N torchrun_multigpu
#PBS -l walltime=01:00:00
#PBS -l select=2:mpiprocs=1:ncpus=64:ngpus=4:mem=480GB
#PBS -q main
#PBS -j oe
##PBS -k eod

# Load modules
module purge
module load nvhpc cuda cray-mpich conda

## Load the conda environment
conda activate evidential


#########################################
# Determine the number of nodes:
if [[ -z "${PBS_NODEFILE}" ]]; then
    echo "PBS_NODEFILE is not set."
    nnodes=1
else
    # Get the number of nodes
    nnodes=$(< $PBS_NODEFILE wc -l)
    echo " nodes : "
    cat $PBS_NODEFILE
fi

echo "number of nodes: $nnodes"

if (( nnodes > 1)); then
    # Get a list of allocated nodes
    nodes=( $( cat $PBS_NODEFILE ) )
    echo nodes: $nodes

    ## --find headnode's IP:
    head_node=${nodes[0]}
    #head_node_ip=$(ssh $head_node hostname -I | awk '{print $1}')
    head_node_ip=$(ssh $head_node hostname -i | awk '{print $1}')

    echo "head_node_ip: $head_node_ip"
    
    # write the head node IP to a file 
    echo $(ssh $head_node hostname -i) > out
fi

#########################################
export LSCRATCH=/glade/derecho/scratch/$USER/
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU

export LOGLEVEL=INFO
export NCCL_DEBUG=INFO


#########################################
# NCCL OFI settings (from Daniel Howard):
export NCCL_SOCKET_IFNAME=hsn
export NCCL_HOME=/glade/u/home/dhoward/work/nccl-ofi-plugin/install
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$NCCL_HOME/plugin/lib:$LD_LIBRARY_PATH

export NCCL_NCHANNELS_PER_NET_PEER=4
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_RDMA_ENABLED_CUDA=1
#export NCCL_DISABLE_IB=1 ## not correct variable in Daniel's email:
export NCCL_IB_DISABLE=1
export NCCL_CROSS_NIC=1
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_OPTIMIZED_MRS=false
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PBH

#########################################

##CMD="torchrun --nodes --nnodes=$nnodes --nproc-per-node=auto --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip main.py"
##echo "Training Command: $CMD"

#-- torchrun launch configuration

LAUNCHER="torchrun "
LAUNCHER+="--nnodes=$nnodes --nproc_per_node=auto --max_restarts 0 "

if [[ "$nnodes" -eq 1 ]]; then
    LAUNCHER+="--standalone "
else
    LAUNCHER+="--rdzv_backend=c10d --rdzv_endpoint=$head_node_ip"
fi

CMD="$LAUNCHER main.py"
echo "Training Command: $CMD"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpiexec -n 2 --ppn 1 --cpu-bind none torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip multigpu_resnet.py 

#mpiexec -n 2 --ppn 4 set_gpu_rank torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip test.py
