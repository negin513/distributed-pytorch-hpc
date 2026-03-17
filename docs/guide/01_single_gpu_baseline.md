# Chapter 1: Single-GPU Baseline

Before we distribute anything, we need a solid single-GPU training script to serve as our starting point. Every distributed strategy in this guide is a modification of this baseline.

## A Complete Training Script

Here's a minimal but complete training loop — a simple U-Net predicting weather on ERA5-like data. We use U-Net here because it's familiar and keeps the focus on distributed training concepts.

> **Note:** State-of-the-art weather models use specialized architectures better suited for global, spherical data:
> - **GraphCast**: Graph Neural Networks on icosahedral mesh 
> - **Pangu-Weather**: 3D Swin Transformers 
> - **FourCastNet v1**: Adaptive Fourier Neural Operators (AFNO) 
> - **FourCastNet v2/3**: Spherical Fourier Neural Operators (SFNO) 
> - **GenCast**: Diffusion models 
> - **Aurora**: 3D Swin Transformer 
>
> The distributed training patterns you learn here apply to all of these.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# 1. Device
device = torch.device("cuda:0")


# 2. Synthetic ERA5-like Dataset
class ERA5Dataset(Dataset):
    """
    Simulates ERA5 reanalysis data for weather prediction.
    
    Input: atmospheric state at time t (multiple variables × pressure levels)
    Target: atmospheric state at time t+6h
    
    Real ERA5 data shape at 0.25° resolution:
    - 721 latitude × 1440 longitude
    - Variables: temperature, humidity, geopotential, wind (u, v)
    - Pressure levels: 13 levels (1000 hPa to 50 hPa)
    """
    def __init__(self, num_samples=1000, num_variables=5, num_levels=13, 
                 lat=721, lon=1440):
        self.num_samples = num_samples
        self.channels = num_variables * num_levels  # e.g., 65 channels
        self.lat = lat
        self.lon = lon
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Input: atmospheric state at time t
        x = torch.randn(self.channels, self.lat, self.lon)
        # Target: atmospheric state at time t+6h
        y = torch.randn(self.channels, self.lat, self.lon)
        return x, y


# 3. Simple U-Net Model
class SimpleUNet(nn.Module):
    """
    A basic U-Net encoder-decoder for demonstration purposes.
    
    This is NOT representative of SOTA weather architectures — it's chosen
    for familiarity so we can focus on distributed training patterns.
    """
    def __init__(self, in_channels=65, out_channels=65, base_dim=64):
        super().__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, base_dim)
        self.enc2 = self._block(base_dim, base_dim * 2)
        self.enc3 = self._block(base_dim * 2, base_dim * 4)
        
        # Bottleneck
        self.bottleneck = self._block(base_dim * 4, base_dim * 8)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_dim * 8, base_dim * 4, kernel_size=2, stride=2)
        self.dec3 = self._block(base_dim * 8, base_dim * 4)
        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=2, stride=2)
        self.dec2 = self._block(base_dim * 4, base_dim * 2)
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=2, stride=2)
        self.dec1 = self._block(base_dim * 2, base_dim)
        
        # Output
        self.out = nn.Conv2d(base_dim, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)


# 4. Create dataset and dataloader
# Using reduced resolution for single-GPU demo (full resolution requires distributed)
train_dataset = ERA5Dataset(
    num_samples=1000,
    num_variables=5,      # T, q, z, u, v
    num_levels=13,        # pressure levels
    lat=181,              # ~1° resolution (downsampled from 0.25°)
    lon=360,
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=4,         # Small batch due to large spatial dimensions
    shuffle=True,
    num_workers=4, 
    pin_memory=True,
)

# 5. Model
model = SimpleUNet(in_channels=65, out_channels=65, base_dim=64)
model = model.to(device)

# 6. Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# 7. Loss function
def latitude_weighted_mse(pred, target):
    """
    MSE weighted by cosine of latitude.
    Accounts for grid cell area differences on a sphere.
    """
    lat = torch.linspace(90, -90, pred.shape[-2], device=pred.device)
    weights = torch.cos(torch.deg2rad(lat))
    weights = weights.view(1, 1, -1, 1)  # (1, 1, lat, 1)
    weights = weights / weights.mean()   # normalize
    
    return (weights * (pred - target) ** 2).mean()

# 8. Training loop
model.train()
for epoch in range(10):
    epoch_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = latitude_weighted_mse(output, target)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f}")
```

## What Goes on GPU Memory (VRAM) in Training?
When training a deep learning model, the GPU memory (VRAM) acts as a high-speed workspace where several distinct components must coexist. If the total memory required by these components exceeds your VRAM capacity, you will encounter the **CUDA Out of Memory (OOM)** error.

During training, four specific components compete for your GPU's memory:
```
┌─────────────────────────────────────────────────────────────┐
│                  GPU Memory (80 GB A100)                    │
│                                                             │
│  ┌─────────────────┐  Model Parameters                      │
│  │   ~400 MB       │  (SimpleUNet: ~100M params             │
│  │                 │   × 4 bytes = 400 MB)                  │
│  ├─────────────────┤                                        │
│  │   ~400 MB       │  Gradients                             │
│  │                 │  (same size as params)                 │
│  ├─────────────────┤                                        │
│  │   ~800 MB       │  Optimizer State                       │
│  │                 │  (AdamW: 2× params for                 │
│  │                 │   momentum + variance)                 │
│  ├─────────────────┤                                        │
│  │   ~20+ GB       │  Activations    ← THE BOTTLENECK       │
│  │                 │  (saved for backward pass,             │
│  │                 │   scales with spatial resolution       │
│  │                 │   AND batch size)                      │
│  └─────────────────┘                                        │
│                                                             │
│  Weather/Climate models have HUGE activations because:      │
│  • High resolution grids (721×1440 at 0.25°)                │
│  • Many channels (variables × pressure levels)              │
│  • Deep networks with skip connections                      │
└─────────────────────────────────────────────────────────────┘
```

### Why Weather/Climate AI Models Are Memory-Hungry

Weather and climate AI models are significantly more memory-intensive than most deep learning models used in computer vision or natural language processing. This is because the atmosphere and Earth system are high-dimensional physical systems that must be represented across space, vertical structure, multiple physical variables, and time. Together, these factors produce extremely large tensors and intermediate activations during training and inference.

- To capture fine-scale weather features, models often use high-resolution grids. For example, ERA5 reanalysis data at 0.25° resolution has a grid of 721 latitudes × 1440 longitudes, resulting in over 1 million spatial points per variable per level. 

- The atmosphere must be modeled in the vertical dimension as well. For example, ERA5 resolves the atmosphere with up to 137 vertical levels from the surface to ~80 km, making atmospheric data inherently three-dimensional.

- Instead of just 3 channels like RGB images, weather models often include dozens of physical variables such as temperature, pressure, humidity, wind components, and geopotential height. Some atmospheric datasets include **tens of variables across multiple pressure levels**, further increasing tensor size.

- Forecasting models typically ingest multiple past timesteps to learn atmospheric dynamics, introducing an additional **time dimension**.

Together, these factors produce extremely large input tensors, often with dimensions like:
```
[batch, time, variables, levels, lat, lon]
```

For example, a single input sample of ERA-5 at 0.25° resolution with 5 variables at 13 pressure levels and 8 surface variables would have, sample size of:

| Dimension        | Size    | Notes                                |
|------------------|---------|--------------------------------------|
| Latitude         | 721     |       |
| Longitude        | 1440    |     |
| Pressure levels  | 13     |  subset of pressure levels |
| Variables/level  | 5       | T, u, v, z, q                        |
| Surface variables| 8       | t2m, u10, v10, msl, tcwv, …         |
| **Total channels** | **73** | (13 levels × 5 vars) + 8 surface    |

```
Single input state:  73 × 721 × 1440 = ~75.8 million values
In float32:          75.8M × 4 bytes  ≈ 303 MB per sample
```
This 303 MB is just the input. During training, neural networks must also store intermediate activations for every layer so that gradients can be computed during backpropagation. These activations are often significantly larger than the input tensor itself, especially in deep convolutional or transformer-based architectures.


#### Activations Are the Real Memory Bottleneck
During training, the GPU must store the output of every intermediate layer (activations) to calculate gradients during the backward pass.

- **Intermediate layer activations** expand the channel dimension significantly. If your model has a hidden dimension of 512 (common in Transformers or Graph Neural Networks), that 303MB input expands to 303 MB × (512/73) ≈ 2.1 GB per layer. With 10 layers, that's already 21 GB of activations.

- **Autoregressive rollouts** multiply the problem further. Weather models are trained to predict multiple future timesteps (e.g., 4 steps × 6h = 24h). To backpropagate through the full rollout, activations from *every step* must be retained in memory, multiplying total activation memory by the rollout length.

- **Spectral transforms** in models like FourCastNet v2/v3 compute spherical harmonic transformsat each layer, producing full spectral coefficient tensors that add further overhead on top of the spatial activations.

- **Attention mechanisms** in transformer-based models (Pangu-Weather, Aurora) store attention maps that scale quadratically with sequence length. On a 721 × 1440 grid, even windowed attention (e.g., 3D Swin Transformer) generates substantial intermediate tensors.

To put this in perspective:
```
Input:                     ~303 MB per sample
Activations (single step): ~20 GB per sample (depending on architecture)
With 4-step rollout:       ~80  GB per sample
```

!!! tip "In Geoscientific AI, activations -- not model parameters -- are the bottleneck. You can have a "small" model (low parameter count) that is still impossible to train on a single GPU because the spatial activations are too massive to fit in VRAM."


## The Three Walls
At some point, your single-GPU training hits one of three limits:

### Wall 1: Training Too Slow
Your model fits on one GPU, but training on 40 years of ERA5 (approx. 5 Petabytes) would take months and you cannot increase batch sizes because you hit OOM. 
**Solution:** Split data across GPUs and train in parallel (DDP).

### Wall 2: Data Too Large (Spatial)
Your input data is too large for a single GPU -- this is especially common in weather/climate. For example, a single 0.1° global sample won't even fit in memory!
**Solution:** Split the input data spatially across GPUs (Domain Parallelism).

### Wall 3: Model Too Large
Your model's parameters + gradients + optimizer state exceed GPU memory. Large foundation models for weather (Aurora, Prithvi) or hybrid physics-ML models (NeuralGCM) can exceed even 80 GB A100s. In this case, shard the model across GPUs (FSDP, Tensor Parallelism, Pipeline Parallelism) or use memory-efficient techniques like gradient checkpointing.


## Other Weather/Climate Specific Considerations

### Latitude Weighting

Grid cells near the poles are smaller than those at the equator. Loss functions should weight by `cos(latitude)`:
```python
def latitude_weighted_mse(pred, target, lat):
    weights = torch.cos(torch.deg2rad(lat))
    return (weights * (pred - target) ** 2).mean()
```

### Temporal Autoregressive Rollout
Weather models are often trained autoregressively — the model's prediction at t+6h becomes the input for t+12h:

```python
# Autoregressive training (simplified)
state = initial_state
for step in range(num_steps):
    next_state = model(state)
    loss += criterion(next_state, target_states[step])
    state = next_state  # Use prediction as next input
```

### Physical Constraints

Unlike general AI, weather models often must satisfy physical laws:

* **Conservation:** Mass and energy should not be "created" by the network.
* **Periodic Boundaries:** The model must understand that the far-right of the grid (longitude) connects back to the far-left.

## What's Next?

Before jumping into code changes, Chapter 2 gives you the conceptual map of all distributed strategies — what each one splits and when to use it.

**Next:** [Chapter 2 — Why Distributed?](02_why_distributed.md)