#Problem Description
# 
# Implement L2 Normalization for a 2D tensor. L2 normalization is a technique where each row of the input tensor is normalized by the Euclidean (L2) norm of its elements.
# 
# The formula for L2 Normalization is:
# $$
# \text{y} = \frac{x}{\sqrt{\sum x_i^2}}
# $$
# where the sum of squared values is computed across the second dimension (D) for each element in the first dimension (B).
# 
# ## Input:
# - Tensor $\text{X}$ of shape $(\text{B}, \text{D})$ (input data)
# 
# ## Output:
# - Tensor $\text{Y}$ of shape $(\text{B}, \text{D})$ (normalized data)
# 
# ## Notes:
# - For numerical stability, you may need to add a small epsilon $\epsilon = 10^{-10}$ to the denominator to avoid division by zero.
# - After normalization, the L2 norm of each row should be approximately 1.0.
# 
# ## Test Case Sizes
# 
# - B=128, D=4096
# - B=256, D=4096
# - B=128, D=8192
# - B=256, D=8192
# - B=128, D=16384


import triton
import triton.language as tl

BLOCK_SIZE = 1024

@triton.jit
def l2_norm_kernel(
    X_ptr,
    Y_ptr,
    D,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * D
    acc = 0.0
    for tile_start in range(0, D, BLOCK_SIZE):
        tile_offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        tile_mask = tile_offsets < D
        x = tl.load(X_ptr + row + tile_offsets, mask=tile_mask, other=0.0)
        acc += tl.sum(x*x)
    acc = tl.extra.cuda.libdevice.sqrt(acc) + eps
    for tile_start in range(0, D, BLOCK_SIZE):
        tile_offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        tile_mask = tile_offsets < D
        x = tl.load(X_ptr + row + tile_offsets, mask=tile_mask, other=0.0)
        tl.store(Y_ptr + row + tile_offsets, x / acc, mask=tile_mask)
    

# Note: X, Y are all float16 device tensors
def solution(X, Y, B: int, D: int):
    eps = 1e-10
    grid = (B,)
    l2_norm_kernel[grid](
        X,
        Y,
        D,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
