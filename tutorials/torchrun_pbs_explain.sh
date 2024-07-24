#!/bin/bash

## This script explains how to run a multi-GPU job on Derecho using torchrun and MPI.
## from a simple example to a more complex examples.

#PBS -A SCSG0001
#PBS -N mpi_torchrun
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
export LSCRATCH=/glade/derecho/scratch/negins/

export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
##export OMP_NUM_THREADS=16
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU

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
#NCCL_SHM_DISABLE


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


#########################################
## mpiexec what does it do with hello world!
mpiexec -n 2 --ppn 1 echo "helloworld!"

## mpiexec print command we created above:
mpiexec -n $nnodes --cpu-bind none echo $CMD

## mpiexec print host name 
mpiexec -n $nnodes --cpu-bind none python  print_hostname.py # print IP address of all nodes

mpiexec -n $nnodes --cpu-bind none python print_hostinfo.py # print IP address of all nodes

#########################################
## mpiexec + torchrun --> print hostname + num of gpu/node
## --nproc-per-node should be number of GPUs/node : auto did not detect correctly on 2nd node. 
#mpiexec -n 2 --cpu-bind none torchrun --nnodes=2 --nproc-per-node=auto --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip main_print_hostname.py

## It either needs --nproc-per-node to set to number of GPUs or explicitly set CUDA_VISIBLE_DEVICES

## solution 1 : 
mpiexec -n 2 --cpu-bind none torchrun --nnodes=$nnodes --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip print_hostinfo.py

## solution 2: 
echo ("after setting cuda visible devices:")
CUDA_VISIBLE_DEVICES=0,1,2,3 mpiexec -n $nnodes --cpu-bind none torchrun --nnodes=$nnodes --nproc-per-node=auto --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip print_hostinfo.py

#########################################
# torchrun_backend needs CUDA_VISIBLE_DEVICES to find GPUs:
CUDA_VISIBLE_DEVICES=0,1,2,3  mpiexec -n $nnodes --cpu-bind none torchrun --nnodes=$nnodes --nproc-per-node=auto --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip test_torchrun.py


### -- Things that did not work:

# -- master_address and master_port did not work:
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   mpiexec -n 2 --cpu-bind none torchrun --nnodes=2 --nproc-per-node=4 --master-addr=$head_node_ip --master-port=8888 print_hostinfo.py  

## -- set_gpu_rank did not work:
# see set_gpu_rank as a provided module --> sets GPUs to be unique when using MPIs
# Rory said we might not have to use mpiexec/torchrun once we run that command
# set_gpu_rank did not work with torchrun because CUDA_VISIBLE_DEVICES is not set correctly for torchrun

## -- Running torchrun alone does not work:
#torchrun --nnodes=1:2 --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip test.py 
#mpiexec -n 2 --ppn 4 --no-transfer set_gpu_rank torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip test.py
#mpiexec -n 2 --ppn 1 torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip test_rank.py
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpiexec -n 2 --ppn 1 --cpu-bind none torchrun --nnodes=2 --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip main.py
#mpiexec -n 2 --ppn 4 set_gpu_rank torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip test.py
#mpiexec -n 8 -ppn 4 get_local_rank torchrun --nnodes 2 --nproc_per_node 4 mp.py
#mpiexec -np 2 -hosts $head_node,$(echo ${nodes[@]} | tr ' ' ',') \
