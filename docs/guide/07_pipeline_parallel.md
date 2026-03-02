# Chapter 7: Pipeline Parallel (PP)

Pipeline Parallelism splits a model **by depth (or model layers)**  different GPUs own
different layers. GPU 0 runs layers 1-20, GPU 1 runs layers 21-40, and
so on. 

This is ideal for very deep models that don't fit on one GPU, but it can be used for any model with many layers.
![Pipeline Parallelism illustration](images/pipe_fairscale.png)


Note that pipeline parallelism can increase the total latency for each request because of communication between different pipeline stages.

Pipeline parallelism is often used in combination with other strategies — for example, you might use TP to split each layer across 4 GPUs, and then PP to split the depth across 4 more GPUs, for a total of 16 GPUs.


## Pipeline Bubble Problem

In a naive pipeline, because each GPU depends on the output of the previous one, some devices may be idle at times, which means resource underutilization. To reduce these idle periods, the input batch can be split into smaller microbatches.

Each microbatch is processed through the pipeline, allowing different stages to work on different microbatches simultaneously. This technique helps to keep all GPUs busy and reduces the idle time (or "pipeline bubbles"), improving overall efficiency. However, it also increases memory usage because activations for multiple microbatches must be stored at once.

### Micro-Batching Reduces Bubbles

Micro-batching allows the pipeline to be filled with multiple micro-batches, reducing idle time. 
Each micro-batch is processed through the pipeline, allowing different stages to work on different micro-batches simultaneously. This technique helps to keep all GPUs busy and reduces the idle time (or "pipeline bubbles"), improving overall efficiency. However, it also increases memory usage because activations for multiple micro-batches must be stored at once.

The solution: split each mini-batch into **micro-batches** and pipeline
them. While GPU 0 processes micro-batch 2, GPU 1 processes micro-batch 1:

```
GPipe schedule (4 stages, 4 micro-batches):

Time ──────────────────────────────────────────────────────►
GPU 0: [ F₁ ][ F₂ ][ F₃ ][ F₄ ]          [ B₄ ][ B₃ ][ B₂ ][ B₁ ]
GPU 1:       [ F₁ ][ F₂ ][ F₃ ][ F₄ ]    [ B₄ ][ B₃ ][ B₂ ][ B₁ ]
GPU 2:             [ F₁ ][ F₂ ][ F₃ ][ F₄ ][ B₄ ][ B₃ ][ B₂ ][ B₁ ]
GPU 3:                   [ F₁ ][ F₂ ][ F₃ ][ F₄ ][ B₄ ][ B₃ ][ B₂ ][ B₁ ]

F = forward    B = backward

Bubble fraction = (stages - 1) / micro-batches = 3/4 = 75% ... still bad
```

More micro-batches → smaller bubble:

```
With 16 micro-batches: bubble = 3/16 = 19%
With 32 micro-batches: bubble = 3/32 = 9%
```

## Typical Implementations of Pipeline Parallelism
There are four common implementations of pipeline parallelism, each with different tradeoffs:
- GPipe: all forwards, then all backwards (high memory)
- PipeDream: introduces asynchronous execution to reduce bubbles (but can cause staleness)
- PipeDream-2BW: Optimizes memory and communication by using 2 weight buffers -- 2-Backward-Weight that reduces staleness
- PipeDream Flush (1F1B): implements the 1F1B (One Forward, One Backward) scheduling strategy to reduce pipeline bubbles and improve efficiency.

### GPipe

All forwards first, then all backwards. Simple but requires storing
activations for all micro-batches simultaneously (high memory), and has a large bubble (idle time) between forward and backward passes.

The illustration below shows the forward and backward passes for 4 stages and 4 micro-batches. The forward pass processes all micro-batches sequentially, followed by the backward pass, which also processes all micro-batches sequentially. This results in a bubble fraction of 75%, meaning that 75% of the time, some GPUs are idle.

![GPipe Pipeline Parallelism with 4 devices and 4 microbatches](https://www.researchgate.net/publication/362249737/figure/fig3/AS:1181989436702720@1658819650017/llustration-of-Pipeline-Parallelism-in-GPipe-with-4-devices-and-4-microbatches-Image.ppm)

```

### 1F1B (One Forward, One Backward)

Interleaves forward and backward passes. Each stage starts backward as
soon as possible, reducing peak activation memory and bubble time. This is the most efficient schedule in practice.

The image below illustrates the 1F1B schedule for 4 stages and 4 micro-batches in PipeDream. In this schedule, as soon as the first micro-batch completes its forward pass on GPU 0, it immediately starts its backward pass while GPU 0 begins processing the second micro-batch. This interleaving continues, allowing for a much smaller bubble fraction of 12.5%, meaning that only 12.5% of the time, some GPUs are idle.

![1F1B microbatch scheduling in PipeDream](https://www.researchgate.net/publication/362249737/figure/fig4/AS:1181989436690441@1658819650284/llustration-of-1F1B-microbatch-scheduling-in-PipeDream-Image-based-on-48.png)

1F1B is preferred in practice because it uses less memory with the same
bubble overhead.

## How Stages Communicate

Pipeline parallelism uses **point-to-point send/recv** between adjacent
stages:

```
Forward:
GPU 0 (layers 1-10)  ──send activations──►  GPU 1 (layers 11-20)
                                              ──send activations──►  GPU 2

Backward:
GPU 2  ──send gradients──►  GPU 1
                             ──send gradients──►  GPU 0
```

This is different from DDP/FSDP (all-reduce, all-gather) — PP
communication is between **pairs** of GPUs, not all GPUs at once.

## Manual Pipeline Splitting

The simplest approach is manual splitting; i.e. you decide which layers go on which GPU:

```python
class Stage0(nn.Module):
    """Layers 0-9, runs on GPU 0."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*[layer() for _ in range(10)])

    def forward(self, x):
        return self.layers(x)

class Stage1(nn.Module):
    """Layers 10-19, runs on GPU 1."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*[layer() for _ in range(10)])

    def forward(self, x):
        return self.layers(x)

# Place on different GPUs
stage0 = Stage0().to("cuda:0")
stage1 = Stage1().to("cuda:1")

# Forward: stage0 → transfer → stage1
x = stage0(input.to("cuda:0"))
x = x.to("cuda:1")  # explicit transfer
output = stage1(x)
```

This approach makes the data flow explicit but requires manual send/recv
for backward and micro-batch scheduling.

## PyTorch Pipeline API

For production use, PyTorch provides `torch.distributed.pipelining`:

```python
from torch.distributed.pipelining import SplitPoint, pipeline, ScheduleGPipe

# Define where to split the model
pipe = pipeline(
    model,
    mb_args=(micro_batch,),
    split_spec={
        "layers.10": SplitPoint.BEGINNING,  # start stage 1 here
        "layers.20": SplitPoint.BEGINNING,  # start stage 2 here
    },
)

# Choose a schedule
schedule = ScheduleGPipe(pipe, n_microbatches=8)

# Run the pipeline
output = schedule.step(input)
```

## Bubble Fraction Formula

The key metric for pipeline parallelism efficiency:

```
bubble_fraction = (num_stages - 1) / num_micro_batches
```

| Stages | Micro-batches | Bubble |
|--------|--------------|--------|
| 2 | 8 | 12.5% |
| 4 | 8 | 37.5% |
| 4 | 16 | 18.8% |
| 4 | 32 | 9.4% |
| 8 | 32 | 21.9% |

Rule of thumb: use at least 4× as many micro-batches as stages.

## When to Use PP

**Use PP when:**
- Your model has many layers and is too deep for one GPU
- The layers are roughly uniform in size (balanced stages)
- You can use enough micro-batches to keep the bubble small

**Prefer other strategies when:**
- Individual layers are too large (use TP instead)
- Your model easily fits with FSDP (simpler to implement)
- You have very few layers (not enough to split meaningfully)


## What's Next?

So far we've split data (DDP), parameters (FSDP), weight matrices (TP),
and layers (PP). But what about the input itself? When sequences are too
long for one GPU, Sequence Parallelism splits the sequence dimension.

**Next:** [Chapter 8 — Sequence Parallel](08_sequence_parallel.md)
