# Problem Description
#
# Compute the Mean Squared Error (MSE) loss between predictions and targets:
# $$
# \text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
# $$
#
# where $y_i$ represents the predictions, $\hat{y}_i$ represents the targets,
# and $N$ is the total number of elements.
#
# ## Input:
# - Tensor `predictions` of arbitrary shape $S_1 \times S_2 \times \cdots \times S_n$
# - Tensor `targets` of the same shape as predictions
# - `shape`: Array containing the dimensions of the input tensors
# - `ndim` ($n$): Number of dimensions in the input tensors
#
# ## Output:
# - Scalar `output` containing the mean squared error loss
#
# ## Notes:
# - The input tensors are stored in row-major order
# - The MSE loss is a scalar value that represents the average squared difference
# - This implementation should handle tensors of arbitrary shapes
#
# ## Test Case Sizes
#
# - shape=(4096, 4096)
# - shape=(8192, 8192)
# - shape=(512, 512, 512)
# - shape=(64, 64, 64, 64)
# - shape=(32, 32, 32, 32, 32)


import triton
import triton.language as tl

BLOCK_SIZE = 1024


@triton.jit
def mse_loss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    acc = 0.0

    pred = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    diff = pred - targets
    res = tl.sum(diff * diff)
    tl.atomic_add(output_ptr, res)


def solution(predictions, targets, output, shape: list[int], ndim: int):
    n_elements = 1
    for s in shape:
        n_elements *= s

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    mse_loss_kernel[grid](
        predictions,
        targets,
        output,
        int(n_elements),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    output /= n_elements
