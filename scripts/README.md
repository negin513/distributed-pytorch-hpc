# Scripts — Distributed Training Examples

Each subdirectory demonstrates a different parallelism strategy with working
code and a PBS job script for Derecho.

| Dir | Strategy | Description | PBS Script |
|-----|----------|-------------|------------|
| `01_data_parallel_ddp/` | DDP | Gradient all-reduce across replicated models | `torchrun_multigpu_ddp.sh` |
| `02_fully_sharded_fsdp/` | FSDP | Shard params, grads, optimizer across GPUs | `run_fsdp.sh` |
| `03_tensor_parallel_tp/` | TP | Split weight matrices across GPUs | `run_tensor_parallel.sh` |
| `04_pipeline_parallel_pp/` | PP | Split model layers across GPUs (GPipe, 1F1B) | `run_pipeline_parallel.sh` |
| `05_sequence_parallel_sp/` | SP | Split sequence dimension for LayerNorm/Dropout | `run_sequence_parallel.sh` |
| `06_hybrid_parallelism/` | TP + FSDP | TP within nodes, FSDP across nodes | `run_hybrid.sh` |
| `07_domain_parallel_shardtensor/` | Domain | Split spatial dimensions (halo exchange) | `run_domain_parallel.sh` |

## Quick Start

```bash
# Submit any example via PBS
qsub scripts/01_data_parallel_ddp/torchrun_multigpu_ddp.sh

# Or run interactively on a GPU node
cd scripts/01_data_parallel_ddp
mpiexec -n 4 --ppn 4 --cpu-bind none python multinode_ddp_basic.py
torchrun --standalone --nproc_per_node=4 multinode_ddp_basic.py
```
