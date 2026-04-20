#Problem Description
# 
# Perform element-wise addition of two vectors:
# $$
# c_i = a_i + b_i
# $$
# 
# ## Input
# - Vectors $a$ and $b$ of length $N$
# 
# ## Output
# - Vector $c$ of length $N$ containing the element-wise sum


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import triton
import triton.language as tl

# Note: d_input1, d_input2, d_output are all float16 device tensors
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(output_ptr + offsets, out, mask=mask)
    
def solution(d_input1, d_input2, d_output, n: int):
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    add_kernel[grid](d_input1, d_input2, d_output, n, BLOCK_SIZE=1024)
    