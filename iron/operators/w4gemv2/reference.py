# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference for the signed-w4 w4gemv2 operator.

Quantization scheme (the engine ABI, differs from the upstream
fused_dequant_gemv reference): per-group-of-32 SYMMETRIC int4 —
nibble in [-8, 7] stored two's-complement packed 2-per-byte (low nibble
first), scale = amax/7 rounded to bf16, dequant w = bf16(nibble) * scale.
The kernel's int4 unpack sign-extends, so no zero-point term exists.

`quantize_and_pack` is shared by the test (random weights) and the model
weight importer (real safetensors): it takes a float (M, K) tensor and
returns the packed DDR buffer plus the dequantized bf16 weights.
"""

import numpy as np
import torch
from ml_dtypes import bfloat16


def quantize_and_pack(W, group_size=32, m_input=16, cols=8):
    """Quantize float tensor W (M, K) to signed w4 and pack for w4gemv2.

    Returns (packed uint8 buffer, W_dequant bf16 torch tensor).
    """
    M, K = W.shape
    assert K % group_size == 0, "K must be a multiple of group_size"
    assert M % cols == 0, "M must be a multiple of cols"
    rows_per_col = M // cols
    assert rows_per_col % m_input == 0, "rows_per_col must be a multiple of m_input"

    num_groups_per_row = K // group_size

    Wg = torch.from_numpy(np.ascontiguousarray(W)).to(torch.float32)
    Wg = Wg.reshape(M * num_groups_per_row, group_size)
    amax = Wg.abs().amax(dim=1, keepdim=True)
    # amax/7 keeps the symmetric range [-8,7] representable on both sides;
    # groups that are exactly zero quantize to nibble 0 / scale 0.
    scale = (amax / 7.0).to(torch.bfloat16).to(torch.float32)
    q = torch.where(amax == 0, torch.zeros_like(Wg), torch.round(Wg / scale))
    q = torch.clamp(q, -8, 7).to(torch.int8)
    W_dequant = (q * scale).to(torch.bfloat16).reshape(M, K)

    # Pack into the tile-based DDR layout: tiles for column 0 first, then
    # column 1, ...; each tile [m_input*K/2 nibble bytes | m_input scale bf16].
    packed_bytes_per_tile = m_input * K // 2 + m_input * num_groups_per_row * 2
    tiles_per_col = rows_per_col // m_input
    packed = np.zeros(cols * tiles_per_col * packed_bytes_per_tile, dtype=np.uint8)

    q_np = q.numpy().reshape(M, num_groups_per_row, group_size)
    scale_np = (
        scale.to(torch.bfloat16).view(torch.uint16).numpy().reshape(M, num_groups_per_row)
    )
    for col in range(cols):
        for tile_idx in range(tiles_per_col):
            row_start = col * rows_per_col + tile_idx * m_input
            tile_offset = (col * tiles_per_col + tile_idx) * packed_bytes_per_tile
            rows = q_np[row_start : row_start + m_input]  # (m_input, groups, 32)
            lo = rows[:, :, 0::2].astype(np.uint8) & 0x0F
            hi = rows[:, :, 1::2].astype(np.uint8) & 0x0F
            packed_tile = (lo | (hi << 4)).reshape(m_input, K // 2)
            packed[
                tile_offset : tile_offset + m_input * K // 2
            ] = packed_tile.reshape(-1)
            s = scale_np[row_start : row_start + m_input]  # (m_input, groups) u16
            off = tile_offset + m_input * K // 2
            packed[off : off + m_input * num_groups_per_row * 2] = s.reshape(-1).view(
                np.uint8
            )

    return packed, W_dequant


def generate_golden_reference(M=2048, K=2048, group_size=32, m_input=16, cols=8, seed=42):
    """Random weights (signed, in [-1, 1)) + random x, packed and golden."""
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
