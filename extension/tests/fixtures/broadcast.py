"""Private regression fixture for operation filtering and source selection."""

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
