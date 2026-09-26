# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Reference for the universal (self-describing-block) w4gemvu operator.

Quantization is the same signed-int4 ABI as w4gemv2 (per-group-32
symmetric, scale = amax/7 bf16, nibbles two's-complement low-first).
The DDR layout is v3 COMPACT BLOCKS (P10) — v2 padded every tile to
13840 B (3x DDR waste at K=2048: qkv/o/gate_up/lm_head streamed 3x
their live bytes; P9's biggest gap item). One block = one fifo element:

  [0 .. n*tile_stride)  n = 6144/K tiles back to back (K=2048 -> 3,
                       K=6144 -> 1), each tile 16-byte ALIGNED (the
                       aie::load_v int4 stream cannot start misaligned —
                       fingerprinted P10: tile@4616 read garbage while
                       tiles@0/9232 were exact):
    [0 .. m*K/2)       row-major nibbles (row r at r*K/2)
    [ .. +8)           junk hole
    [ .. +m*(K/32)*2)  bf16 scales
  [ .. 13880)          pad (48 B at K=6144, 8 B per K=2048 tile)
  [13880..13884)       K as u32 (little-endian)
  [ .. 13888)          reserved

The in-tile anchors match what the compiled kernel actually reads (peano
drops the +8 on the int4 weight stream; scales and the K load keep
theirs — see w4gemvu.cc). The core calls the kernel 3x per block; calls
past the live tile count write zero rows, so K=6144 outputs carry 2/3
zero rows interleaved (12-row groups, real rows first) and K=2048 is
dense. acquire(1) per block keeps the kernel's single-pointer contract.
"""

import numpy as np
import torch
from ml_dtypes import bfloat16

ELEM = 13888
K_MAX = 6144
CALLS_PER_BLOCK = 3  # static core-loop count (kernel skips tiles >= n)
BLOCKS_PER_B = 2     # blocks served per B slot (single-BD fills)


def tiles_per_block(k):
    return K_MAX // k  # 3 (K=2048) or 1 (K=6144)


def blocks_per_col(M, K, m_input=4, cols=8):
    """Fifo blocks one column streams (M is the PADDED row count)."""
    return (M // cols // m_input) // tiles_per_block(K)


def quantize_and_pack(W, group_size=32, m_input=4, cols=8):
    """Quantize float tensor W (M, K) to signed w4 and pack compact blocks.

    M must satisfy the v3 ABI: tiles divide the block packing AND blocks
    are even (one B slot serves two blocks): K=2048 -> M % 192 == 0,
    K=6144 -> M % 64 == 0. Pad with zero rows upstream (they quantize to
    q=0/scale=0 and produce zero outputs).

    Returns (packed uint8 buffer, W_dequant bf16 torch tensor).
    """
    M, K = W.shape
    assert K in (2048, 6144), "w4gemvu variants cover K in {2048, 6144}"
    assert K % group_size == 0
    assert M % cols == 0 and (M // cols) % m_input == 0
    n = tiles_per_block(K)
    assert (M // cols // m_input) % n == 0, \
        f"K={K}: tiles must pack {n}/block (M % {cols*m_input*n} == 0)"
    assert blocks_per_col(M, K, m_input, cols) % 2 == 0, \
        "blocks per column must be even (B slot serves two blocks)"

    num_groups_per_row = K // group_size
    tile_bytes = 8 + m_input * K // 2 + m_input * num_groups_per_row * 2
    tile_stride = (tile_bytes + 15) & ~15  # 16B-aligned (load_v stream)
    assert n * tile_stride <= ELEM - 16, "block must hold the tiles + header"

    Wg = torch.from_numpy(np.ascontiguousarray(W)).to(torch.float32)
    Wg = Wg.reshape(M * num_groups_per_row, group_size)
    amax = Wg.abs().amax(dim=1, keepdim=True)
    scale = (amax / 7.0).to(torch.bfloat16).to(torch.float32)
    q = torch.where(amax == 0, torch.zeros_like(Wg), torch.round(Wg / scale))
    q = torch.clamp(q, -8, 7).to(torch.int8)
    W_dequant = (q * scale).to(torch.bfloat16).reshape(M, K)

    tiles_per_col = M // cols // m_input
    blocks = tiles_per_col // n
    packed = np.zeros(cols * blocks * ELEM, dtype=np.uint8)

    q_np = q.numpy().reshape(M, num_groups_per_row, group_size)
    scale_np = (
        scale.to(torch.bfloat16).view(torch.uint16).numpy().reshape(M, num_groups_per_row)
    )
    import struct

    k_le = struct.pack("<I", K)
    for col in range(cols):
        for b in range(blocks):
            off = (col * blocks + b) * ELEM
            packed[off + ELEM - 8 : off + ELEM - 4] = np.frombuffer(k_le, dtype=np.uint8)
            for i in range(n):
                t = b * n + i
                row_start = col * (M // cols) + t * m_input
                toff = off + i * tile_stride
                # In-tile layout: nibbles at 0 (row-major), 8-byte hole,
                # scales — the anchors the compiled kernel reads (peano
                # drops the +8 on the int4 stream; indexed loads keep it).
                rows = q_np[row_start : row_start + m_input]
                lo = rows[:, :, 0::2].astype(np.uint8) & 0x0F
                hi = rows[:, :, 1::2].astype(np.uint8) & 0x0F
                nibbles = (lo | (hi << 4)).reshape(m_input, K // 2)
                packed[toff : toff + m_input * K // 2] = nibbles.reshape(-1)
                s = scale_np[row_start : row_start + m_input]
                o = toff + m_input * K // 2 + 8
                packed[o : o + m_input * num_groups_per_row * 2] = s.reshape(-1).view(np.uint8)

    return packed, W_dequant


def output_rows(M, K):
    """Rows the C drain writes (zero-padding past the live tiles)."""
    return M * CALLS_PER_BLOCK // tiles_per_block(K)  # M (K2048) / 3M (K6144)


def unshuffle_output(c_rows, M, K):
    """Device C rows -> real (M,) output (K=6144 rows sit at 12t..12t+4)."""
    if K == 2048:
        return c_rows[:M].clone()
    out = torch.zeros(M, dtype=c_rows.dtype)
    n = tiles_per_block(K)
    stride = CALLS_PER_BLOCK * 4
    for t in range(M // 4):
        out[t * 4 : (t + 1) * 4] = c_rows[t * stride : t * stride + 4]
    return out


def shuffle_output(real, M, K):
    """Real (M,) rows -> the device C-row layout (inverse of unshuffle)."""
    if K == 2048:
        return real.clone()
    raw = torch.zeros(output_rows(M, K), dtype=real.dtype)
    stride = CALLS_PER_BLOCK * 4
    for t in range(M // 4):
        raw[t * stride : t * stride + 4] = real[t * 4 : (t + 1) * 4]
    return raw


def generate_golden_reference(M, K, group_size=32, m_input=4, cols=8, seed=42):
    """Random signed weights + deterministic x, packed and golden."""
    torch.manual_seed(seed)
    W = (torch.rand(M, K, dtype=torch.float32) * 2 - 1).numpy()
    x = (torch.rand(K, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)

    packed, W_dequant = quantize_and_pack(W, group_size, m_input, cols)

    ref = (W_dequant.to(torch.float32) @ x.to(torch.float32)).to(torch.bfloat16)

    return {
        "packed_weights": packed,
        "x": x,
        "output": ref,
    }
