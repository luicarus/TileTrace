"""Educational FlashAttention forward: one contiguous fp16 batch/head.

Open this file in TileTrace; no Triton installation or GPU is needed to inspect it.
Launch grid for execution: (triton.cdiv(N_CTX, BLOCK_M),). Q/K/V/O are contiguous
[N_CTX, HEAD_DIM] fp16 tensors. Default scale = 1/sqrt(HEAD_DIM); update SM_SCALE
when changing HEAD_DIM. Block dimensions are powers of two and at least 16.

Each program owns BLOCK_M query rows and streams BLOCK_N key/value rows. Online
softmax keeps only tile-sized scores and a running row maximum, denominator,
and output accumulator. CAUSAL masks future keys. The finite mask sentinel is
intended for finite fp16 inputs at the default scale, not arbitrary magnitudes.
This is a readable forward example, not a tuned benchmark or a backward kernel.
Algorithm reference: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
"""

import triton
import triton.language as tl


@triton.jit
def flash_attention_forward(
    Q, K, V, O,
    N_CTX: tl.constexpr = 64,
    HEAD_DIM: tl.constexpr = 32,
    BLOCK_M: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 32,
    SM_SCALE: tl.constexpr = 0.1767766952966369,
    CAUSAL: tl.constexpr = True,
):
    # Program 0: query rows [0,16); program 1: [16,32); etc.
    query_rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    head_cols = tl.arange(0, HEAD_DIM)
    query_offsets = query_rows[:, None] * HEAD_DIM + head_cols[None, :]
    q = tl.load(Q + query_offsets, mask=query_rows[:, None] < N_CTX, other=0)

    row_max = tl.full((BLOCK_M,), -1.0e30, tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    # Defaults give two KV blocks: [0,32) and [32,64).
    for start_n in tl.static_range(0, N_CTX, BLOCK_N):
        key_rows = start_n + tl.arange(0, BLOCK_N)
        kv_offsets = key_rows[:, None] * HEAD_DIM + head_cols[None, :]
        k = tl.load(K + kv_offsets, mask=key_rows[:, None] < N_CTX, other=0)
        v = tl.load(V + kv_offsets, mask=key_rows[:, None] < N_CTX, other=0)
        k_transposed = tl.trans(k)
        scores = tl.dot(q, k_transposed) * SM_SCALE

        valid_keys = key_rows[None, :] < N_CTX
        if CAUSAL:
            valid_keys = valid_keys & (query_rows[:, None] >= key_rows[None, :])
        scores = tl.where(valid_keys, scores, -1.0e30)

        next_max = tl.maximum(row_max, tl.max(scores, axis=1))
        correction = tl.exp(row_max - next_max)
        probabilities = tl.exp(scores - next_max[:, None])
        next_sum = row_sum * correction + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None]
        probabilities_fp16 = probabilities.to(tl.float16)
        accumulator = tl.dot(probabilities_fp16, v, accumulator)
        row_max = next_max
        row_sum = next_sum

    normalized = accumulator / row_sum[:, None]
    output = normalized.to(tl.float16)
    tl.store(O + query_offsets, output, mask=query_rows[:, None] < N_CTX)
