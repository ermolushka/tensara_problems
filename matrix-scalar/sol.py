# Problem Description

# Multiply every element of a matrix by a scalar value:

# $$
# C[i][j] = A[i][j] \cdot s
# $$

# ## Input:
# - Matrix $A$ of size $M \times N$ stored in row-major order
# - Scalar $s$ (float)

# ## Output:
# - Matrix $C$ of size $M \times N$

# ## Test Case Sizes

# - M=1024, N=1024
# - M=2048, N=2048
# - M=4096, N=4096
# - M=8192, N=8192


import triton
import triton.language as tl

BLOCK_SIZE = 1024

@triton.jit
def matrix_scalar_kernel(
    A_ptr,
    C_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * n_elements
    for tile_start in range(0, n_elements, BLOCK_SIZE):
        tile_offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        tile_mask = tile_offsets < n_elements
        a = tl.load(A_ptr + offsets + tile_offsets, mask=tile_mask, other=0.0)
        tl.store(C_ptr + offsets + tile_offsets, a * scalar, mask=tile_mask)


def solution(input_matrix, scalar: float, output_matrix, n: int):
    grid = (n,)
    matrix_scalar_kernel[grid](
        input_matrix,
        output_matrix,
        scalar,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )

