#Problem Description
# 
# Perform the Leaky ReLU (Leaky Rectified Linear Unit) activation function on an input matrix:
# $$
# \text{C}[i][j] = \max(\alpha \cdot \text{A}[i][j], \text{A}[i][j])
# $$
# where $\alpha$ is a small positive constant (e.g. 0.01)
# 
# The Leaky ReLU function is defined as:
# $$
# f(x) = \begin{cases} 
# x & \text{if } x > 0 \\
# \alpha x & \text{if } x \leq 0 
# \end{cases}
# $$
# 
# ## Input:
# - Matrix $\text{A}$ of size $M \times N$ 
# - $\alpha$ value (slope for negative values)
# 
# ## Output:
# - Matrix $\text{C}$ of size $M \times N$
# 
# ## Notes:
# - Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
# - This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/20_LeakyReLU.py)
# 
# ## Test Case Sizes
# 
# - 4096x4096, alpha=0.01
# - 4096x4096, alpha=0.05
# - 4096x4096, alpha=0.1
# - 4096x4096, alpha=0.2
# - 6144x4096, alpha=0.01
# - 6144x4096, alpha=0.05
# - 6144x4096, alpha=0.1
# - 6144x4096, alpha=0.2


import triton
import triton.language as tl

BLOCK_SIZE = 1024

@triton.jit
def leaky_relu_kernel(
    input_ptr,
    output_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    a = tl.where(a >= 0, a, alpha * a)
    tl.store(output_ptr + offsets, a, mask=mask)



# Note: input, output are all float16 device tensors
def solution(input, alpha: float, output, n: int, m: int):
    n_elements = n * m
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    leaky_relu_kernel[grid](
        input,
        output,
        alpha,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
