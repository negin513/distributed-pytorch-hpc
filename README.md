# Multi-Node Multi-GPU PyTorch Training on NCAR's Derecho

Developed by : [Negin Sobhani](https://github.com/negin513)

This repostory contains example workflows with for executing multi-node, multi-GPU machine learning training using PyTorch on NCAR's HPC Supercomputers (i.e. Derecho). 

While this code is written to run directly on [Derecho](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/) GPU nodes, it can be adapted for other GPU HPC machines. 
Each [Derecho](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/) node has 4 NVIDIA A100 GPUs. The examples in this repository demonstrate how to train a model on multiple GPUs across multiple nodes using `torch.distribtued` and `torchrun`.

The goal of this repository is to provide a starting point for researchers who want to scale their PyTorch training to multiple GPUs and nodes on NCAR's HPC systems.

In this repository, we provide examples of how to train a ResNet model on multiple GPUs across multiple nodes using PyTorch's Distributed Data Parallel (DDP) library and Fully Sharded Data Parallel (FSDP) library.

## What is DDP?

Distributed Data Parallel (DDP) is a PyTorch library that allows you to train your model on multiple GPUs across multiple nodes. DDP is a wrapper around PyTorch's `torch.nn.DataParallel` module, which is used to parallelize the training of a model across multiple GPUs on a single node. DDP extends this functionality to multiple nodes, allowing you to scale your training to hundreds of GPUs.

To learn more about DDP, check out the [official PyTorch DDP documentation](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).



## What is FSDP (Fully Sharded Data Parallelism)?
FSDP is a PyTorch library that allows you to train very large models that don't fit on a single GPU across multiple GPUs and nodes. FSDP shards the model parameters across multiple GPUs and nodes, allowing you to train very large models that don't fit on a single GPU. FSDP is a more advanced version of DDP that is specifically designed for training very large models on multiple GPUs and nodes. 
Please see the image below for a comparison of DDP and FSDP:
https://openmmlab.medium.com/its-2023-is-pytorch-s-fsdp-the-best-choice-for-training-large-models-fe8d2848832f


## What should I use? (DP vs. DDP vs. FSDP)

- DP (Data Parallelism): Use this when you have a small model that fits on a single GPU and you want to train it on multiple GPUs on a single node. It is the simplest and most common form of parallelism in PyTorch.

- DDP (Distributed Data Parallelism): Use this when you have a large model that doesn't fit on a single GPU and you want to train it on multiple GPUs across multiple nodes.

- FSDP (Fully Sharded Data Parallelism): Use this when you have a very large model that doesn't fit on a single GPU and you want to train it on multiple GPUs across multiple nodes. FSDP is a more advanced version of DDP that shards the model parameters across multiple GPUs and nodes, allowing you to train very large models that don't fit on a single GPU.

Here is a summary table of the different parallelism strategies:

| **Strategy** | **Number of Nodes** | **Number of GPUs per Node** | **Launch Method**                              |
|--------------|---------------------|-----------------------------|------------------------------------------------|
| **DP (Data Parallel)** | 1                   | >=1                    | N/A only need using `torch.nn.DataParallel`  in the script|
| **DDP (Distributed Data Parallel)** | >=1            | >=1                    | `torchrun` when `N=1`  <br> `mpirun` + `torchrun` when `N>1` |
| **FSDP (Fully Sharded Data Parallel)** | Multiple            | Multiple                    | `mpirun` + `torchrun` with setup for sharding |


## PyTorch PBS Commands Explained (`mpirun` and `torchrun`)

In order to use `torchrun` or `distributed.launch` to run distributed training (DDP or FSDP) on two nodes, you need ssh into each node, find the IP, and run the following command:

```bash
# the master node ---> e.g. 104.171.200.62
torchrun \
    --nproc_per_node=2 --nnodes=4 --node_rank=0 \
    --rdzv-backend=c10d --rdzv-endpoint=104.171.200.62\
    main.py

# On worker node (different IP)
torchrun \
    --nproc_per_node=2 --nnodes=4 --node_rank=1 \
    --rdzv-backend=c10d --rdzv-endpoint=104.171.200.62\
    main.py
```

In the above lines:

- `--nodes` define the number of nodes.
- `--nproc_per_node` define the number of GPUs per node.
- `--node_rank` define the rank of the node which is `0` for the master node and `1` for the worker node.

Although the above lines would work nicely on each node, one need start interactive jobs and then ssh into each node and run the command. To avoid this, we can use MPI to run the command on all nodes at once. 

MPI (Message Passing Interface) is a standard for parallel computing that allows you to run the same command on multiple nodes at once.  For example the following command using `mpiexec` would run the same command on all nodes (two nodes) at once:

```bash
## hello world!
mpiexec -n 2 --ppn 1 echo "helloworld!"
```

Or in the following example,  the `mpiexec` command would run the same `torchrun` command on all nodes (two nodes) at once. Next, `torchrun` runs the python code `tutorials/print_hostinfo.py`: 

```bash
mpiexec -n $nnodes --cpu-bind none \
    torchrun --nnodes=$nnodes --nproc-per-node=auto \
    --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip tutorials/print_hostinfo.py
```

In the above line:

- `$nnodes` is the number of nodes.
- `cpu-bind none` is used to avoid binding the CPU to the GPU, which would hurt the GPU performance. 
- `--nproc-per-node=auto` is used to automatically detect the number of GPUs per node with the help of `torchrun`. The user can either specify the number of GPUs per node or let `torchrun` detect it automatically from the environment variables.

- `--rdzv-backend=c10d` is used to specify the rendezvous backend. 
- `--rdzv-endpoint=$head_node_ip` is used to specify the IP of the head node.



With the advancement in CUDA applications and GPU clusters, libraries like NCCL (NVIDIA Collective Communication Library) provide faster inter-GPU communication primitives that are topology-aware, leveraging technologies such as RDMA via RoCE or InfiniBand. NCCL integrates easily into MPI applications, with MPI serving as the frontend for launching the parallel job and NCCL as the backend for heavy communication.



## Example Workflows

Example 1) Multi-GPU ResNet Training on Derecho using DDP `/scripts/torchrun_multigpu_pbs.sh`

In this example, we demonstrate how to train a ResNet model on multiple GPUs across multiple nodes using PyTorch's Distributed Data Parallel (DDP) library. 

To submit this job on Derecho, you can use the following command:

```bash
qsub scripts/torchrun_multigpu_pbs.sh
```

Example 2) Multi-GPU ResNet Training on Derecho using FSDP with PBS `/scripts/torchrun_multigpu_fsdp.sh`

In this example, we demonstrate how to train a ResNet model on multiple GPUs across multiple nodes using PyTorch's Fully Sharded Data Parallel (FSDP) library.

To submit this job on Derecho, you can use the following command:


``` bash
qsub scripts/torchrun_multigpu_fsdp.sh
```





## Resources


