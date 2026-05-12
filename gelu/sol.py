#Problem Description
# 
# Perform the GELU (Gaussian Error Linear Unit) activation function on an input matrix:
# $$
# C[i][j] = \text{GELU}(A[i][j])
# $$
# 
# The GELU function is defined as:
# $$
# \text{GELU}(x) = x \cdot \Phi(x)
# $$
# 
# where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution. 
# 
# A common approximation for GELU is:
# $$
# \text{GELU}(x) \approx 0.5x \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715x^3)))
# $$
# 
# ## Input:
# - Matrix $A$ of size $M \times N$ containing floating-point values
# 
# ## Output:
# - Matrix $C$ of size $M \times N$ containing the GELU activation values
# 
# ## Notes:
# - Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
# - You should implement the approximation formula for GELU defined above
# - GELU is commonly used in modern transformer-based neural networks like BERT and GPT
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
def gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    cubic = a * a * a
    a = 0.5 * a * (1 + tl.extra.cuda.libdevice.tanh(tl.extra.cuda.libdevice.sqrt(2/3.141592653589793) * (a + 0.044715*cubic)))
    tl.store(output_ptr + offsets, a, mask=mask)


# Note: input, output are all float16 device tensors
def solution(input, output, n: int, m: int):
    n_elements = n * m
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    gelu_kernel[grid](
        input,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
