#Problem Description
# 
# Perform the Tanh activation function on an input matrix:
# $$
# C[i][j] = \text{tanh}(A[i][j])
# $$
# 
# The Tanh function is defined as:
# $$
# \text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
# $$
# 
# ## Input:
# - Matrix $A$ of size $M \times N$ containing floating-point values
# 
# ## Output:
# - Matrix $C$ of size $M \times N$ containing the Tanh activation values
# 
# ## Notes:
# - Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
# - This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/22_Tanh.py)
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
def tanh_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x = (tl.exp(x) - tl.exp(-x)) / (tl.exp(x) + tl.exp(-x))
    tl.store(output_ptr + offsets, x, mask=mask)


# Note: input, output are all float16 device tensors
def solution(input, output, n: int, m: int):
    n_elements = n * m
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    tanh_kernel[grid](
        input,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
