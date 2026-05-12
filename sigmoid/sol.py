#Problem Description
# 
# Perform the Sigmoid activation function on an input matrix:
# $$
# C[i][j] = \sigma(A[i][j])
# $$
# 
# The Sigmoid function is defined as:
# $$
# \sigma(x) = \frac{1}{1 + e^{-x}}
# $$
# 
# ## Input:
# - Matrix $A$ of size $M \times N$ containing floating-point values
# 
# ## Output:
# - Matrix $C$ of size $M \times N$ containing the Sigmoid activation values
# 
# ## Notes:
# - Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
# - This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/21_Sigmoid.py)
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
def sigmoid_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    x = 1 / (1 + tl.exp(-x))
    tl.store(output_ptr + offsets, x, mask=mask)


# Note: input, output are all float16 device tensors
def solution(input, output, n: int, m: int):
    n_elements = n * m
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    sigmoid_kernel[grid](
        input,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
