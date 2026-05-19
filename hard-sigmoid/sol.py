#Problem Description
#
# Perform the Hard Sigmoid activation function on an input matrix:
# $$
# C[i][j] = \text{hard\_sigmoid}(A[i][j])
# $$
#
# The Hard Sigmoid function is defined as:
# $$
# \text{hard\_sigmoid}(x) = \begin{cases}
# 0 & \text{if } x \leq -3 \\
# 1 & \text{if } x \geq 3 \\
# \frac{x + 3}{6} & \text{otherwise}
# \end{cases}
# $$
#
# ## Input:
# - Matrix $A$ of size $M \times N$ containing floating-point values
#
# ## Output:
# - Matrix $C$ of size $M \times N$ containing the Hard Sigmoid activation values
#
# ## Notes:
# - Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
# - The Hard Sigmoid function is a piecewise linear approximation of the standard Sigmoid function
#
# ## Test Case Sizes
#
# - 4096x4096
# - 6144x4096
# - 4096x7168
# - 4096x8192
# - 8192x8192


import triton
import triton.language as tl

BLOCK_SIZE = 1024

@triton.jit
def hard_sigmoid_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * n_elements
    for tile_start in range(0, n_elements, BLOCK_SIZE):
        tile_offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        tile_mask = tile_offsets < n_elements
        x = tl.load(input_ptr + offsets + tile_offsets, mask=tile_mask, other=0.0)
        x = tl.where(x <= -3, 0, tl.where(x >= 3, 1, (x+3)/6))
        tl.store(output_ptr + offsets + tile_offsets, x, mask=tile_mask)


def solution(input, output, n: int, m: int):
    grid = (n,)
    hard_sigmoid_kernel[grid](
        input,
        output,
        m,
        BLOCK_SIZE=BLOCK_SIZE,
    )
