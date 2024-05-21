# Multi-Node Multi-GPU PyTorch Training on NCAR's Derecho

Developed by : Negin Sobhani

This repostory contains example workflows with for executing multi-node, multi-GPU machine learning training using PyTorch on NCAR's HPC Supercomputers (i.e. Derecho). 

Each Derecho node has 4 NVIDIA V100 GPUs. The examples in this repository demonstrate how to train a model on multiple GPUs across 1 node and multiple nodes using PyTorch's Distributed Data Parallel (DDP) library.

## What is DDP?

Distributed Data Parallel (DDP) is a PyTorch library that allows you to train your model on multiple GPUs across multiple nodes. DDP is a wrapper around PyTorch's `torch.nn.DataParallel` module, which is used to parallelize the training of a model across multiple GPUs on a single node. DDP extends this functionality to multiple nodes, allowing you to scale your training to hundreds of GPUs.

To learn more about DDP, check out the [official PyTorch documentation](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

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

## What is FSDP (Fully Sharded Data Parallelism)?
FSDP is a PyTorch library that allows you to train very large models that don't fit on a single GPU across multiple GPUs and nodes. FSDP shards the model parameters across multiple GPUs and nodes, allowing you to train very large models that don't fit on a single GPU. FSDP is a more advanced version of DDP that is specifically designed for training very large models on multiple GPUs and nodes. 
Please see the image below for a comparison of DDP and FSDP:
https://openmmlab.medium.com/its-2023-is-pytorch-s-fsdp-the-best-choice-for-training-large-models-fe8d2848832f


## What should I use?

- DP (Data Parallelism): Use this when you have a small model that fits on a single GPU and you want to train it on multiple GPUs on a single node. It is the simplest and most common form of parallelism in PyTorch.

- DDP (Distributed Data Parallelism): Use this when you have a large model that doesn't fit on a single GPU and you want to train it on multiple GPUs across multiple nodes.

- FSDP (Fully Sharded Data Parallelism): Use this when you have a very large model that doesn't fit on a single GPU and you want to train it on multiple GPUs across multiple nodes. FSDP is a more advanced version of DDP that shards the model parameters across multiple GPUs and nodes, allowing you to train very large models that don't fit on a single GPU.

Here is a summary table of the different parallelism strategies:

| **Strategy** | **Number of Nodes** | **Number of GPUs per Node** | **Launch Method**                              |
|--------------|---------------------|-----------------------------|------------------------------------------------|
| **DP (Data Parallel)** | 1                   | >=1                    | Directly in script using `torch.nn.DataParallel`  |
| **DDP (Distributed Data Parallel)** | >1            | >=1                    | `mpirun` + `torchrun`  |
| **FSDP (Fully Sharded Data Parallel)** | Multiple            | Multiple                    | `mpirun` + `torchrun` with setup for sharding |

 
## Getting Started

Example workflows for executing multi-node, multi-GPU machine learning training using PyTorch on NCAR's HPC Supercomputers.
