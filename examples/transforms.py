"""Open this file in the viewer. Static analysis does not import Triton."""

import triton
import triton.language as tl


@triton.jit
def broadcast_demo(BLOCK_M: tl.constexpr = 2, BLOCK_N: tl.constexpr = 4):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_column = rows[:, None]
    column_row = cols[None, :]
    matrix = row_column + column_row
    transposed = tl.trans(matrix)
    row_sum = tl.sum(matrix, axis=1)
    row_sum_column = row_sum[:, None]
    centered = matrix - row_sum_column


@triton.jit
def reshape_demo(BLOCK: tl.constexpr = 8):
    vector = tl.arange(0, BLOCK)
    matrix = tl.reshape(vector, (2, 4))
    transposed = matrix.trans()
    kept_axis = tl.max(transposed, axis=0, keep_dims=True)


@triton.jit
def softmax_demo(X, N, BLOCK: tl.constexpr = 8):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=-float("inf"))
    maximum = tl.max(x, axis=0)
    centered = x - maximum
    numerator = tl.exp(centered)
    denominator = tl.sum(numerator, axis=0)
    probabilities = numerator / denominator
