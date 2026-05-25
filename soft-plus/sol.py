# Problem Description
#
# Perform the Softplus activation function on an input matrix:
# C[i][j] = softplus(A[i][j])
#
# The Softplus function is defined as:
# softplus(x) = ln(1 + e^x)
#
# ## Input:
# - Matrix A of size M x N
#
# ## Output:
# - Matrix C of size M x N
#
# ## Notes:
# - Both matrices are stored in row-major order
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
def softplus_kernel(
    a_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offsets < n_elements
    
    x = tl.load(a_ptr + row_offsets, mask=row_mask, other=0.0)
    res = tl.log(1 + tl.exp(x))
    tl.store(c_ptr + row_offsets, res, mask=row_mask)


def solution(a, c, m: int, n: int):
    n_elements = m * n
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    softplus_kernel[grid](
        a,
        c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
