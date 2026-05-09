#Problem Description
# 
# Perform the ReLU (Rectified Linear Unit) activation function on an input matrix:
# $$
# C[i][j] = \max(0, A[i][j])
# $$
# 
# The ReLU function is defined as:
# $$
# f(x) = \begin{cases} 
# x & \text{if } x > 0 \\
# 0 & \text{if } x \leq 0 
# \end{cases}
# $$
# 
# ## Input:
# - Matrix $A$ of size $M \times N$ containing floating-point values
# 
# ## Output:
# - Matrix $C$ of size $M \times N$ containing the ReLU activation values
# 
# ## Notes:
# - Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
# - This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/19_ReLU.py)
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
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BLOCK_SIZE = 1024

@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(input_ptr + offsets, mask=mask)
    a = tl.maximum(a, 0.0)
    tl.store(output_ptr + offsets, a, mask=mask)


# Note: input, output are all float16 device tensors
def solution(input, output, n: int, m: int):
    n_elements = n * m
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    relu_kernel[grid](
        input,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    