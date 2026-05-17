
Implement L1 Normalization for a 2D tensor.

Normalize each row of the input by dividing by its L1 norm (sum of absolute values). More formally, compute:

$$
\text{y}_i = \frac{x_i}{\sum_j |x_{i,j}|}
$$

for each sample $i$ independently.

## Input:
- Tensor $\text{X}$ of shape $(\text{B}, \text{N})$ where $\text{B}$ = batch size and $\text{N}$ = number of features

## Output:
- Tensor $\text{Y}$ with the same shape as input (L1-normalized data)

## Notes:
- Normalization is computed along the feature dimension (dimension 1) for each sample independently.
- After normalization, the sum of absolute values of each row equals 1.

## Test Case Sizes

- shape=(1024, 1024)
- shape=(1024, 4096)
- shape=(2048, 8192)
- shape=(512, 16384)
