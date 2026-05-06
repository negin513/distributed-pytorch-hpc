#!/usr/bin/env python3
"""
01_why_splitting_fails.py - Single-GPU demonstration

This script runs on ONE GPU and demonstrates WHY naive spatial splitting
of data gives incorrect convolution results at the split boundary.

No distributed setup needed. Just run:
    python 01_why_splitting_fails.py

The Problem:
    A 3x3 convolution needs to see neighboring pixels. When you split an
    image in half along the height dimension, the convolution at the boundary
    (rows 511-512) can't see neighbors that live in the other half.

    This is the fundamental problem that domain parallelism (ShardTensor)
    solves automatically.
"""

import torch
import torch.nn as nn

torch.manual_seed(42)

# ============================================================
# Step 1: Create a full image and a convolution
# ============================================================
full_image = torch.randn(1, 8, 1024, 1024)  # [batch, channels, height, width]
conv = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)

# Compute the "ground truth" output on the full image
full_output = conv(full_image)

# ============================================================
# Step 2: Split the image in half along height (dim=2)
#   This simulates what 2 GPUs would each hold.
# ============================================================
left_half = full_image[:, :, :512, :]   # rows 0-511
right_half = full_image[:, :, 512:, :]  # rows 512-1023

# Apply the SAME convolution to each half independently
left_output = conv(left_half)
right_output = conv(right_half)
naive_output = torch.cat([left_output, right_output], dim=2)

# ============================================================
# Step 3: Compare - they DON'T match!
# ============================================================
print("=== Naive Splitting (INCORRECT) ===")
print(f"Full output shape:  {full_output.shape}")
print(f"Naive output shape: {naive_output.shape}")
print(f"Outputs match? {torch.allclose(full_output, naive_output)}")

# Find WHERE they disagree
diff = torch.abs(full_output - naive_output)
_, _, h_locs, _ = torch.where(diff > 1e-6)
bad_rows = torch.unique(h_locs)
print(f"Rows with errors: {bad_rows.tolist()}")
print(f"  -> Only rows 511 and 512, right at the split boundary!\n")

# ============================================================
# Step 4: Fix it manually with halo exchange
#   Each half needs 1 extra row from its neighbor (for a 3x3 kernel,
#   the "halo" is 1 pixel on each side of the split).
# ============================================================
# The left half needs 1 row from the right half's start
halo_for_left = right_half[:, :, 0:1, :]   # row 512
# The right half needs 1 row from the left half's end
halo_for_right = left_half[:, :, -1:, :]    # row 511

# Pad each half with the halo
padded_left = torch.cat([left_half, halo_for_left], dim=2)    # 513 rows
padded_right = torch.cat([halo_for_right, right_half], dim=2) # 513 rows

# Convolve the padded halves, then trim back to original size
left_output_fixed = conv(padded_left)[:, :, :-1, :]   # drop last row
right_output_fixed = conv(padded_right)[:, :, 1:, :]   # drop first row

fixed_output = torch.cat([left_output_fixed, right_output_fixed], dim=2)

print("=== Manual Halo Exchange (CORRECT) ===")
print(f"Outputs match? {torch.allclose(full_output, fixed_output)}")
print()

# ============================================================
# Summary
# ============================================================
print("KEY TAKEAWAY:")
print("  Splitting spatial data across GPUs requires 'halo exchange' -")
print("  sharing boundary pixels between neighbors - for convolutions")
print("  (and many other ops) to produce correct results.")
print()
print("  ShardTensor automates this for you, including the backward pass")
print("  for gradient computation. See 02_shardtensor_conv.py next.")
