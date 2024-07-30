This directory includes some example scripts for running PyTorch training on multiple GPUs across multiple nodes using `torchrun`.

Important scripts:
```
scripts/
├── torchrun_multigpu_pbs.sh # Performance Test for Multi-GPU ResNet Training on Derecho using DDP
|-- main.py # Main script for training ResNet model (downloads data automatically) includes timing and stats for performance testing
|-- multigpu_resnet.py # ResNet model (simple test no timing)
```


Running `torchrun_multigpu_pbs.sh` will submit a job that trains a ResNet model on multiple GPUs across multiple nodes using PyTorch's Distributed Data Parallel (DDP) library with and without NCCL OFI plugin. 

This will give you timing results for the training jobs in `reresnet_benchmark.log` 

Preliminary results show that the NCCL OFI plugin can provide a significant speedup for multi-GPU training on Derecho.

```
with OFI plugin: 
w/o OFI plugin : nccl 2-19-3: Average epoch time: 4.805356439917978 sec.
w OFI plugin     nccl 2-19-3: Average epoch time: 2.795084637824935 sec.
```
