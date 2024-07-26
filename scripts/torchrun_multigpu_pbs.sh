#!/bin/bash

## This script is for running a multi-GPU job on Derecho using torchrun and MPI. 
## This script runs 

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
#module load conda intel cray-mpich cuda cudnn
## Load the conda environment
#conda activate evidential
conda activate /glade/work/schreck/conda-envs/holodec
#conda activate /glade/work/schreck/conda-envs/credit
#conda activate torch_cuda

which torchrun

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

export NCCL_SOCKET_IFNAME=hsn # needed for NCCL --> not working without this. 

# no performance improvement with these settings:
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1

# needed to avoid hangs by Josh Romero (NVIDIA):
#export FI_CXI_DISABLE_HOST_REGISTER=1
#export FI_CXI_OPTIMIZED_MRS=false

# -- old settings not sure if they had large impacts:

export NCCL_SHM_DISABLE=1 # probably doesn't matter
export NCCL_IB_DISABLE=1  # probably doesn't matter
export NCCL_CROSS_NIC=1 # should have an impact based on docs but it does not. (Josh Romero said improve perf on non-rail optimized networks)
export NCCL_NCHANNELS_PER_NET_PEER=4
#export NCCL_NET_GDR_LEVEL=PHB
###exexport OMP_NUM_THREADS=16


#export NCCL_HOME=/glade/u/apps/common/23.08/spack/opt/spack/nvhpc/24.1/Linux_x86_64/24.1/comm_libs/12.3/nccl
#export LD_LIBRARY_PATH=$NCCL_HOME/lib:$NCCL_HOME/plugin/lib:$LD_LIBRARY_PATH

DEBUG=0
if [ "$ENABLE_NCCL_OFI" -eq 1 ]; then
    export LSCRATCH=/glade/derecho/scratch/negins/
    export LOGLEVEL=INFO
    export NCCL_DEBUG=INFO
    #export NCCL_DEBUG=TRACE #VERSION
else    
    export NCCL_DEBUG=VERSION
fi

#########################################
ENABLE_NCCL_OFI=0  # Set to 1 to enable, 0 to disable

activate_nccl_ofi() {
  local enable_nccl_ofi=$1

  if [ "$enable_nccl_ofi" -eq 1 ]; then
    #########################################
    # NCCL OFI settings (from Daniel Howard):
    export NCCL_HOME=/glade/u/home/dhoward/work/nccl-ofi-plugin/install
    export LD_LIBRARY_PATH=$NCCL_HOME/lib:$NCCL_HOME/plugin/lib:$LD_LIBRARY_PATH

    export NCCL_NCHANNELS_PER_NET_PEER=4
    export MPICH_GPU_SUPPORT_ENABLED=1
    export MPICH_OFI_NIC_POLICY=GPU
    export MPICH_RDMA_ENABLED_CUDA=1
    export NCCL_IB_DISABLE=1
    export NCCL_CROSS_NIC=1
    export NCCL_NET="AWS Libfabric" # not needed from Negin's opinion
    export NCCL_NET_GDR_LEVEL=PBH
    #export NCCL_DEBUG=INFO
    #########################################
  fi
}

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

which mpiexec
export CUDA_VISIBLE_DEVICES=0,1,2,3


LOGFILE=resnet_benchmark.log

ENABLE_NCCL_OFI=0  # Set to 1 to enable, 0 to disable
activate_nccl_ofi $ENABLE_NCCL_OFI

# -- does not work:
#python -m torch.utils.collect_env
echo "------------------------------------" >> $LOGFILE
echo "ENABLE_NCCL_OFI" $ENABLE_NCCL_OFI >> $LOGFILE
date +"%Y-%m-%d %H:%M:%S" >> $LOGFILE
rm -f snapshot.pt

mpiexec -n 2 --ppn 1 --cpu-bind none torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip main.py 


ENABLE_NCCL_OFI=1
activate_nccl_ofi $ENABLE_NCCL_OFI
echo "ENABLE_NCCL_OFI" $ENABLE_NCCL_OFI >> $LOGFILE

rm -f snapshot.pt

mpiexec -n 2 --ppn 1 --cpu-bind none torchrun --nnodes=$nnodes --nproc-per-node=auto --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip main.py 
date +"%Y-%m-%d %H:%M:%S" >> $LOGFILE

echo "------------------------------------" >> $LOGFILE