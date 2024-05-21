#!/bin/bash
#PBS -A SCSG0001
#PBS -N old
#PBS -l walltime=01:00:00
#PBS -l select=2:mpiprocs=1:ncpus=64:ngpus=4:mem=480GB
#PBS -q main
#PBS -j oe
#PBS -k eod

# see set_gpu_rank as a provided module --> sets GPUs to be unique when using MPIs
# Rory said we might not have to use mpiexec/torchrun once we run that command

# Load modules
module purge
module load nvhpc cray-mpich conda 
ml cuda/11.7.0
ml
conda activate evidential
#conda activate /glade/work/schreck/conda-envs/holodec
#conda activate torch_cuda
#export PATH=$PATH:/glade/u/home/negins/bin
which torchrun
#export NCCL_HOME=/glade/u/apps/common/23.08/spack/opt/spack/nvhpc/24.1/Linux_x86_64/24.1/comm_libs/12.3/nccl

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
export MPICH_GPU_SUPPORT_ENABLED=1
#export NCCL_SOCKET_IFNAME=hsn
#export NCCL_DISABLE_IB=1
export NCCL_DEBUG=INFO

which mpiexec
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
mpiexec -n 2 --ppn 1 --cpu-bind none torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip comm_test.py

#mpiexec -n 2 --ppn 4 set_gpu_rank torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip test.py
