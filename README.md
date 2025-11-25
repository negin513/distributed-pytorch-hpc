# Multi-Node Multi-GPU PyTorch Training on NCAR's Derecho

**Under Development!!!**

Developed by : [Negin Sobhani](https://github.com/negin513)

This repostory contains example workflows with for executing multi-node, multi-GPU machine learning training using PyTorch on NCAR's HPC Supercomputers (i.e. Derecho). 

While this code is written to run directly on [Derecho](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/) GPU nodes, it can be adapted for other GPU HPC machines.

Each [Derecho](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/derecho/) node has 4 NVIDIA A100 GPUs. The examples in this repository demonstrate how to train a model on multiple GPUs across multiple nodes using `torch.distributed` and `torchrun`.

The goal of this repository is to provide a starting point for researchers who want to scale their PyTorch training to multiple GPUs and nodes on Derecho.

## Quick Start
To get started with multi-node, multi-GPU PyTorch training on Derecho, follow these steps:

```bash


```
## Repository Structure

In this repository, you will find the following:

- [`test/`](tests/README.md): This directory contains test scripts for testing performance of nccl with example PBS scripts of running them.  
- [`scripts/`](scripts/README.md): This directory contains example PBS scripts for training multi-node, multi-GPU neural network on Derecho using PyTorch's DDP. It includes timing and stats summary useful for performance analysis.
- `tutorials/`: This directory contains simple scripts for testing torch and nccl installation on Derecho with example PBS scripts of running them and explaining the arguments.
- `environment.yml` : This file contains the conda environment for running the example workflows.


## How to Make the Environment:

To create the conda environment, you can use the following command:

```bash
module load conda
CONDA_OVERRIDE_CUDA=12.1 mamba env create -f environment.yml
conda activate pytorch_cuda_env
```



## 🧩 Parallelism Strategies in PyTorch
Modern deep learning training uses different parallelism strategies depending on model size, GPU memory, and scaling goals. PyTorch provides several built-in mechanisms for scaling across multiple GPUs and nodes.

This repository follows the natural progression of these strategies:


### 1. Distributed Data Parallel (DDP)
Distributed Data Parallel (DDP) is a PyTorch library that allows you to train your model on multiple GPUs across multiple nodes. DDP is a wrapper around PyTorch's `torch.nn.DataParallel` module, which is used to parallelize the training of a model across multiple GPUs on a single node. DDP extends this functionality to multiple nodes, allowing you to scale your training to hundreds of GPUs.

📘 Learn more: **[PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)**



### 2. Fully Sharded Data Parallel (FSDP)
FSDP is a PyTorch library that allows you to train very large models that don't fit on a single GPU across multiple GPUs and nodes. FSDP shards the model parameters across multiple GPUs and nodes, allowing you to train very large models that don't fit on a single GPU. FSDP is a more advanced version of DDP that is specifically designed for training very large models on multiple GPUs and nodes. 

To learn more about FSDP, check out the [official PyTorch FSDP documentation](https://pytorch.org/docs/stable/fsdp.html).

FSDP handles very large models (1B+ parameters).... 



Please see the image below for a comparison of DDP and FSDP:
https://openmmlab.medium.com/its-2023-is-pytorch-s-fsdp-the-best-choice-for-training-large-models-fe8d2848832f


### 3. Tensor Parallelism (TP)
Tensor Parallelism (TP) is a technique used to split the model's tensors across multiple GPUs. This allows for training larger models that don't fit on a single GPU. TP is often used in conjunction with DDP or FSDP to further scale the training of large models.

Layers are split within the model, and each GPU holds only a slice of a layer.
GPUs share the same input batch
Useful for extremely wide layers (Transformers)
Built on PyTorch’s DeviceMesh APIs
Use when: Individual layers are too large for a single GPU.

### 4. Hybrid Parallelism (FSDP + TP)
Combine data parallel sharding with intra-layer model parallelism.
Enables 10B–100B parameter LLMs
Common in GPT-style and LLaMA-style training
Requires careful configuration of parallel dimensions
Use when: Training very large foundation models at scale.



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


| Strategy | Nodes Supported | GPUs per Node | Model Fits on 1 GPU? | What is Split? | Data Split? | Typical Use Case | How to Launch |
|----------|------------------|----------------|-----------------------|----------------|--------------|------------------|----------------|
| **DP (`nn.DataParallel`)** | 1 | ≥1 | Yes | Nothing (threading only) | Yes | Legacy single-node multi-GPU (NOT recommended) | None (in-script) |
| **DDP (`torchrun`)** | 1 to many | ≥1 | Yes | Gradients (all-reduce) | Yes | Standard multi-GPU/multi-node training | `torchrun` (1 node) <br> `mpirun` + `torchrun` (multi-node) |
| **FSDP (Fully Sharded)** | 1 to many | ≥1 | No | Params + Grads + Optim State (sharded) | Yes | Large models (1B+), memory-limited workloads | `mpirun` + `torchrun` |
| **Tensor Parallel (TP)** | 1 to many | ≥2 | No (wide layers) | Model layers (intra-layer tensor shards) | No (all GPUs see same batch) | Transformers with very wide layers (e.g., attention) | `torchrun` + `DeviceMesh` config |
| **Pipeline Parallel (PP)** | 1 to many | ≥1 | No (deep models) | Layers assigned to stages | Yes (per stage) | Very deep models; microbatch pipelines | `torchrun` with pipeline engine |
| **Hybrid (FSDP + TP)** | 1 to many | ≥2 | No (massive models) | Both model + tensor shards | Yes | LLMs (10B–100B), foundation models | `mpirun` + `torchrun` with 2D mesh |



----
## Launching Multi-Node Jobs on Derecho 

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

Although the above lines would work on each node, one need start interactive jobs and then ssh into each node and run the command. To avoid this, we can use **MPI** (Message Passing Interface) to run the command on all nodes at once. 


👉 ** You should not use this method for running distributed training jobs unless you have a specific reason to do so.**

MPI (Message Passing Interface) is a standard for parallel computing that allows you to run the same command on multiple nodes at once.  


For example the following command using `mpiexec` would run the same command on all nodes (two nodes) at once:

```bash
mpiexec -n 4 --ppn 2 echo "helloworld!"
```

In the above line:
- `-n` is the number of ranks.
- `--ppn` is the number of ranks per node.

We can extend this to run `torchrun` on all nodes at once using `mpiexec`.
Here, the `mpiexec` command would run the same `torchrun` command on all nodes (two nodes) at once. Next, `torchrun` runs the python code `tutorials/print_hostinfo.py`: 

```bash
mpiexec -n $nranks --ppn $gpus_per_node tutorials/print_hostinfo.py
```

In the above line:

- `$nranks` is the number of ranks.
- `cpu-bind none` is used to avoid binding the CPU to the GPU, which would hurt the GPU performance. 
- `--nproc-per-node=auto` is used to automatically detect the number of GPUs per node with the help of `torchrun`. The user can either specify the number of GPUs per node or let `torchrun` detect it automatically from the environment variables.

- `--rdzv-backend=c10d` is used to specify the rendezvous backend. 
- `--rdzv-endpoint=$head_node_ip` is used to specify the IP of the head node.


**Key insight: Use --ppn 1 (one MPI rank per node), not one per GPU. torchrun handles GPU spawning.**

### Alternative: Direct MPI Launch (No `torchrun`)

For simpler scripts or when using MPI backend directly, you can directly invoke `mpiexec` to launch your training script without `torchrun`.

```bash
# Set environment variables for torch.distributed
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500

# Launch with MPI (1 rank per GPU)
mpiexec -n 8 --ppn 4 --cpu-bind none python train.py --backend nccl
```


On Derecho, NCCL is recommended for GPU training. For optimal performance with the Slingshot interconnect, use a build with the AWS OFI NCCL plugin (see build instructions in the https://github.com/benkirk/derecho-pytorch-mpi repository).




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

- [Multi node PyTorch Distributed Training Guide For People In A Hurry](https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide#distributed-pytorch-underthehood)
