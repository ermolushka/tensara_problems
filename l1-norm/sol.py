#Problem Description
#
# Implement L1 Normalization for a 2D tensor.
#
# Normalize each row of the input by dividing by its L1 norm (sum of absolute values). More formally, compute:
#
# $$
# \text{y}_i = \frac{x_i}{\sum_j |x_{i,j}|}
# $$
#
# for each sample $i$ independently.
#
# ## Input:
# - Tensor $\text{X}$ of shape $(\text{B}, \text{N})$ where $\text{B}$ = batch size and $\text{N}$ = number of features
#
# ## Output:
# - Tensor $\text{Y}$ with the same shape as input (L1-normalized data)
#
# ## Notes:
# - Normalization is computed along the feature dimension (dimension 1) for each sample independently.
# - After normalization, the sum of absolute values of each row equals 1.
#
# ## Test Case Sizes
#
# - shape=(1024, 1024)
# - shape=(1024, 4096)
# - shape=(2048, 8192)
# - shape=(512, 16384)


import triton
import triton.language as tl

BLOCK_SIZE = 1024

@triton.jit
def l1_norm_kernel(
    X_ptr,
    Y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offsets = pid * N
    acc = 0.0

    for tile_start in range(0, N, BLOCK_SIZE):
        tile_offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        tile_mask = tile_offsets < N

        tile = tl.load(X_ptr + row_offsets + tile_offsets, mask=tile_mask, other=0.0)
        acc += tl.sum(tl.abs(tile))
        
    for tile_start in range(0, N, BLOCK_SIZE):
        tile_offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        tile_mask = tile_offsets < N

        tile = tl.load(X_ptr + row_offsets + tile_offsets, mask=tile_mask, other=0.0)
        tl.store(Y_ptr + row_offsets + tile_offsets, tile/acc, mask=tile_mask)

# Note: X, Y are all float16 device tensors
def solution(X, Y, B: int, N: int):
    grid = (B,)
    l1_norm_kernel[grid](
        X,
        Y,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
