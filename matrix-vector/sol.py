#Problem Description
# 
# Perform multiplication of a matrix and a vector:
# $$
# C[i] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k]
# $$
# 
# ## Input:
# - Matrix $A$ of size $M \times K$
# - Vector $B$ of size $K \times 1$
# 
# ## Output:
# - Vector $C = AB$ of size $M \times 1$
# 
# ## Notes:
# - Matrix $\text{A}$ is stored in row-major order
# - This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/4_Matrix_vector_multiplication_.py)
# 
# ## Test Case Sizes
# 
# - M=4096, K=4096
# - M=6144, K=4096
# - M=7168, K=4096
# - M=8192, K=4096
# - M=9216, K=4096


import triton
import triton.language as tl

BLOCK_SIZE_M = 64
BLOCK_SIZE_K = 64

@triton.jit
def matvec_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask = offsets < m
    # accumulator to store temp data
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    # iterate over K tiles
    for k_start in range(0, k, BLOCK_SIZE_K):
        tile_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        tile_mask = tile_offsets < k
        # load k tile of vector
        k_tile = tl.load(b_ptr + tile_offsets, mask=tile_mask, other=0.0)
        # load k tile of matrix 
        a_tile = tl.load(a_ptr + offsets[:, None] * k + tile_offsets[None, :], mask=mask[:, None] & tile_mask[None, :], other=0.0)
        # multiply and sum over K axis, so we multiply data in a row and sum by the row
        mult = a_tile * k_tile[None, :]
        res = tl.sum(mult, axis=1) # over K dim
        acc += res
    tl.store(c_ptr + offsets, acc, mask=mask)


# Note: input_a, input_b, output_c are all float16 device tensors
def solution(input_a, input_b, output_c, m: int, k: int):
    grid = (triton.cdiv(m, BLOCK_SIZE_M),)
    matvec_kernel[grid](
        input_a,
        input_b,
        output_c,
        m,
        k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    