#Problem Description
# 
# Implement RMS (Root Mean Square) Normalization for a 2D tensor.
# 
# Normalize the input by dividing each element by the root mean square of the features in each sample. More formally, compute:
# 
# $$
# \text{y} = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}
# $$
# 
# where the mean is computed along the feature dimension for each sample in the batch independently. $\epsilon$ is a small value added to the denominator for numerical stability.
# 
# ## Input:
# - Tensor $\text{X}$ of shape $(\text{B}, \text{N})$ is the input data where $\text{B}$ = batch size and $\text{N}$ = number of features
# 
# ## Output:
# - Tensor $\text{Y}$ with the same shape as input (normalized data)
# 
# ## Notes:
# - For each sample, the RMS is calculated over the feature dimension (dimension 1).
# - Use $\epsilon = 10^{-5}$ for numerical stability.
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
def rms_norm_kernel(
    X_ptr,
    Y_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * N
    acc = 0.0

    # we do 2 passes as our feature dimensions
    # if bigger than block size + we need to accumulate
    # statistics for the entire row
    for tile_start in range(0, N, BLOCK_SIZE):
        tile_offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        tile_mask = tile_offsets < N
        tile = tl.load(X_ptr + offsets + tile_offsets, mask=tile_mask, other=0.0)
        acc += tl.sum(tile * tile)
    rms = tl.extra.cuda.libdevice.sqrt(acc / N + eps)
    
    # on the second iteration just divide tile
    # on rms
    for tile_start in range(0, N, BLOCK_SIZE):
        tile_offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        tile_mask = tile_offsets < N
        tile = tl.load(X_ptr + offsets + tile_offsets, mask=tile_mask, other=0.0)

        tl.store(Y_ptr + offsets + tile_offsets, tile / rms, mask=tile_mask)

    
# Note: X, Y are all float16 device tensors
def solution(X, Y, B: int, N: int):
    eps = 1e-5
    grid = (B,)
    rms_norm_kernel[grid](
        X,
        Y,
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
