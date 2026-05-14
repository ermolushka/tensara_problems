#Problem Description
#
# Perform the ELU (Exponential Linear Unit) activation function on an input matrix:
# $$
# \text{C}[i][j] = \begin{cases}
# \text{A}[i][j] & \text{if } \text{A}[i][j] > 0 \\
# \alpha \cdot (e^{\text{A}[i][j]} - 1) & \text{if } \text{A}[i][j] \leq 0
# \end{cases}
# $$
# where $\alpha$ is a positive constant (e.g. 1.0)
#
# The ELU function is defined as:
# $$
# f(x) = \begin{cases}
# x & \text{if } x > 0 \\
# \alpha (e^x - 1) & \text{if } x \leq 0
# \end{cases}
# $$
#
# ## Input:
# - Matrix $\text{A}$ of size $M \times N$
# - $\alpha$ value (scale for negative values)
#
# ## Output:
# - Matrix $\text{C}$ of size $M \times N$
#
# ## Notes:
# - Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
# - This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/31_ELU.py)
#
# ## Test Case Sizes
#
# - 4096x4096, alpha=1.0
# - 4096x4096, alpha=0.5
# - 4096x4096, alpha=2.0
# - 4096x4096, alpha=0.1
# - 6144x4096, alpha=1.0
# - 6144x4096, alpha=0.5
# - 6144x4096, alpha=2.0
# - 6144x4096, alpha=0.1


import triton
import triton.language as tl

BLOCK_SIZE = 1024

@triton.jit
def elu_kernel(
    input_ptr,
    output_ptr,
    alpha: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x = tl.where(x > 0, x, alpha * (tl.exp(x) - 1))
    tl.store(output_ptr + offsets, x, mask=mask)


# Note: input, output are all float16 device tensors
def solution(input, output, n: int, m: int, alpha: float):
    alpha_val = float(alpha.item()) if hasattr(alpha, 'item') else float(alpha)
    n_val = int(n.item()) if hasattr(n, 'item') else int(n)
    m_val = int(m.item()) if hasattr(m, 'item') else int(m)
    n_elements = n_val * m_val
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    elu_kernel[grid](
        input,
        output,
        alpha_val,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
