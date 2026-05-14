
Perform the ELU (Exponential Linear Unit) activation function on an input matrix:
$$
\text{C}[i][j] = \begin{cases} 
\text{A}[i][j] & \text{if } \text{A}[i][j] > 0 \\
\alpha \cdot (e^{\text{A}[i][j]} - 1) & \text{if } \text{A}[i][j] \leq 0 
\end{cases}
$$
where $\alpha$ is a positive constant (e.g. 1.0)

The ELU function is defined as:
$$
f(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha (e^x - 1) & \text{if } x \leq 0 
\end{cases}
$$

## Input:
- Matrix $\text{A}$ of size $M \times N$
- $\alpha$ value (scale for negative values)

## Output:
- Matrix $\text{C}$ of size $M \times N$

## Notes:
- Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order
- This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/31_ELU.py)

## Test Case Sizes

- 4096x4096, alpha=1.0
- 4096x4096, alpha=0.5
- 4096x4096, alpha=2.0
- 4096x4096, alpha=0.1
- 6144x4096, alpha=1.0
- 6144x4096, alpha=0.5
- 6144x4096, alpha=2.0
- 6144x4096, alpha=0.1
