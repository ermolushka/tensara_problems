#Problem Description
#
# Perform RGB to grayscale conversion on an input image using the weighted method:
# $$
# \text{Gray}[i][j] = 0.299 \cdot \text{R}[i][j] + 0.587 \cdot \text{G}[i][j] + 0.114 \cdot \text{B}[i][j]
# $$
#
# This formula accounts for human perception of color, with green contributing most to the intensity perceived by humans.
#
# ## Input:
# - RGB image of size $\text{height} \times \text{width} \times \text{3}$
#
# ## Output:
# - Grayscale image of size $\text{height} \times \text{width}$
#
# ## Notes:
# - The input tensor is in HWC format (height, width, channels)
# - In memory, the tensor is stored in row-major order with interleaved channels (R,G,B,R,G,B,...)
# - Each pixel has 3 channels in RGB order
# - Pixel values are in the range [0, 255] for both input and output
# - The output is a single-channel grayscale image
#
# ## Test Case Sizes
#
# - 512x512
# - 1024x768
# - 1920x1080
# - 3840x2160


import triton
import triton.language as tl

BLOCK_SIZE = 512

@triton.jit
def grayscale_kernel(
    rgb_ptr,
    gray_ptr,
    n_pixels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * n_pixels * 3
    for tile_start in range(0, n_pixels, BLOCK_SIZE):
        tile_offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        tile_mask = tile_offsets < n_pixels
        r = tl.load(rgb_ptr + offsets + tile_offsets * 3, mask=tile_mask, other=0.0)
        g = tl.load(rgb_ptr + offsets + tile_offsets * 3 + 1, mask=tile_mask, other=0.0)
        b = tl.load(rgb_ptr + offsets + tile_offsets * 3 + 2, mask=tile_mask, other=0.0)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        tl.store(gray_ptr + pid * n_pixels + tile_offsets, y, mask=tile_mask)



def solution(rgb_image, grayscale_output, height: int, width: int, channels: int):
    n_pixels = height * width
    grid = (height,)
    grayscale_kernel[grid](
        rgb_image,
        grayscale_output,
        width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
