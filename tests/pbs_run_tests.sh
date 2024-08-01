#!/bin/bash

## This script runs a multi-GPU job on Derecho using torchrun and MPI.
## This script runs speed tests for different communication backends.

#PBS -A SCSG0001
#PBS -N performance_test
#PBS -l walltime=01:00:00
#PBS -l select=2:mpiprocs=1:ncpus=64:ngpus=4
#PBS -q main
#PBS -j oe
##PBS -k eod


# Load modules
module purge
#module load conda nvhpc cray-mpich cuda cudnn
module load conda intel cray-mpich


## Load the conda environment
conda activate /glade/work/negins/conda-envs/pytorch_cuda_env
#conda activate /glade/work/schreck/conda-envs/holodec

which torchrun
which mpiexec

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
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_OPTIMIZED_MRS=false

# -- old settings not sure if they had large impacts:

export NCCL_SHM_DISABLE=1 # probably doesn't matter
export NCCL_IB_DISABLE=1  # probably doesn't matter
export NCCL_CROSS_NIC=1 # should have an impact based on docs but it does not. (Josh Romero said improve perf on non-rail optimized networks)
#export NCCL_NET_GDR_LEVEL=PHB
###exexport OMP_NUM_THREADS=16


#export NCCL_HOME=/glade/u/apps/common/23.08/spack/opt/spack/nvhpc/24.1/Linux_x86_64/24.1/comm_libs/12.3/nccl
#export LD_LIBRARY_PATH=$NCCL_HOME/lib:$NCCL_HOME/plugin/lib:$LD_LIBRARY_PATH


########################################
# Debugging settings:
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
## NCCL OFI Plugin settings:
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
    export NCCL_DEBUG=INFO
    #########################################
  fi
}


#########################################
export CUDA_VISIBLE_DEVICES=0,1,2,3


ENABLE_NCCL_OFI=0  # Set to 1 to enable, 0 to disable
activate_nccl_ofi $ENABLE_NCCL_OFI

#python -m torch.utils.collect_env > collect_env.log -- > this does not work.

echo "------------------------------------" >> benchmark_results.log
echo "ENABLE_NCCL_OFI" $ENABLE_NCCL_OFI >> benchmark_results.log

# send_recv speed tests (nccl vs. gloo):
mpiexec -n 2 --ppn 1 --cpu-bind none torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip send_recv_test.py --backend nccl
mpiexec -n 2 --ppn 1 --cpu-bind none torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip send_recv_test.py --backend gloo


# allreduce speed tests (nccl vs. gloo):
mpiexec -n 2 --ppn 1 --cpu-bind none torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip all_reduce_test.py --backend nccl
mpiexec -n 2 --ppn 1 --cpu-bind none torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip all_reduce_test.py --backend gloo

ENABLE_NCCL_OFI=1
activate_nccl_ofi $ENABLE_NCCL_OFI

echo "ENABLE_NCCL_OFI" $ENABLE_NCCL_OFI >> benchmark_results.log

# send_recv speed tests (nccl vs. gloo):
mpiexec -n 2 --ppn 1 --cpu-bind none torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip send_recv_test.py --backend nccl
mpiexec -n 2 --ppn 1 --cpu-bind none torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip send_recv_test.py --backend gloo


# allreduce speed tests (nccl vs. gloo):
mpiexec -n 2 --ppn 1 --cpu-bind none torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip all_reduce_test.py --backend nccl
mpiexec -n 2 --ppn 1 --cpu-bind none torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip all_reduce_test.py --backend gloo


echo "------------------------------------" >> benchmark_results.log