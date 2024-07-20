#!/bin/bash

#PBS -A SCSG0001
#PBS -N testing_ddp
#PBS -l walltime=01:00:00
#PBS -l select=2:mpiprocs=1:ncpus=64:ngpus=4:mem=480GB
#PBS -q main
#PBS -j oe
##PBS -k eod

# see set_gpu_rank as a provided module --> sets GPUs to be unique when using MPIs
# Rory said we might not have to use mpiexec/torchrun once we run that command

# Load modules
module purge
module load nvhpc cuda cray-mpich conda
conda activate evidential
module list


# Check if PBS_NODEFILE is set
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


CMD="torchrun --nodes --nnodes=$nnodes --nproc-per-node=auto --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip main.py"
echo "Training Command: $CMD"


export LSCRATCH=/glade/derecho/scratch/negins/

export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
##export OMP_NUM_THREADS=16
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1

# NCCL OFI settings
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
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PBH



## mpiexec what does it do with hello world!
mpiexec -n 2 --ppn 1 echo "helloworld!"
mpiexec -n $nnodes --cpu-bind none echo $CMD

## mpiexec print host name 
mpiexec -n $nnodes --cpu-bind none python  print_hostname.py # print IP address of all nodes


## mpiexec + torchrun --> print hostname + num of gpu/node
## --nproc-per-node should be number of GPUs/node : auto did not detect correctly on 2nd node. 
#mpiexec -n 2 --cpu-bind none torchrun --nnodes=2 --nproc-per-node=auto --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip main_print_hostname.py

#CUDA_VISIBLE_DEVICES=0,1,2,3
mpiexec -n 2 --cpu-bind none torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip print_hostinfo.py

#torchrun --nnodes=2 --nproc-per-node=2 --rdzv-backend=c10d  --master_port=1234 khar.py
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=2 --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip main.py
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=2 --nproc-per-node=4 --rdzv-endpoint=$head_node_ip  --master-port=8888 main.py

#torchrun --nnodes=1:2 --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip test.py 
#mpiexec -n 2 --ppn 4 --no-transfer set_gpu_rank torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip test.py
#mpiexec -n 2 --ppn 1 torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip test_rank.py
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpiexec -n 2 --ppn 1 --cpu-bind none torchrun --nnodes=2 --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip main.py
#mpiexec -n 2 --ppn 4 set_gpu_rank torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip test.py
#mpiexec -n 8 -ppn 4 get_local_rank torchrun --nnodes 2 --nproc_per_node 4 mp.py
#mpiexec -np 2 -hosts $head_node,$(echo ${nodes[@]} | tr ' ' ',') \
