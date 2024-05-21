#!/bin/bash
#PBS -A SCSG0001 
#PBS -N torchrun_comm
#PBS -l walltime=01:00:00
#PBS -l select=2:mpiprocs=1:ncpus=64:ngpus=4:mem=480GB
#PBS -q main
#PBS -j oe
#PBS -k eod

# see set_gpu_rank as a provided module --> sets GPUs to be unique when using MPIs
# Rory said we might not have to use mpiexec/torchrun once we run that command

# Load modules
module purge
module load nvhpc cuda cray-mpich conda 
conda activate evidential

# Get a list of allocated nodes
nodes=( $( cat $PBS_NODEFILE ) )
head_node=${nodes[0]}
head_node_ip=$(ssh $head_node hostname -i | awk '{print $1}')

echo Head Node IP: $head_node_ip
echo $(ssh $head_node hostname -i) 2>&1 |tee out

export LSCRATCH=/glade/derecho/scratch/$USER/
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU

export LOGLEVEL=INFO
#export NCCL_DEBUG=INFO

export NCCL_SOCKET_IFNAME=hsn
export NCCL_HOME=/glade/u/home/dhoward/work/nccl-ofi-plugin/install
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$NCCL_HOME/plugin/lib:$LD_LIBRARY_PATH

export NCCL_NCHANNELS_PER_NET_PEER=4
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_RDMA_ENABLED_CUDA=1
export NCCL_DISABLE_IB=1
export NCCL_CROSS_NIC=1
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_OPTIMIZED_MRS=false
#export NCCL_NET="AWS Libfabric"
#export NCCL_NET_GDR_LEVEL=PBH


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpiexec -n 2 --ppn 1 --cpu-bind none torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip multigpu_resnet.py 

#mpiexec -n 2 --ppn 4 set_gpu_rank torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip test.py
