# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference for the universal (self-describing-tile) w4gemvu operator.

Quantization is the same signed-int4 ABI as w4gemv2 (per-group-32
symmetric, scale = amax/7 bf16, nibbles two's-complement low-first).
The DDR layout differs: every tile is a SELF-DESCRIBING PADDED SLOT —
layout v2, built for the peano anchor quirk (see w4gemvu.cc):

  [0 .. m*K/2)      row-major nibbles (row r at r*K/2)
  [ .. +8)          junk hole
  [ .. +m*(K/32)*2) bf16 scales
  [ .. 13832)       pad
  [13832..13836)    K as u32
  [ .. 13840)       reserved

so one element geometry serves every shape and the fills stay
element-aligned (notes §14). acquire(1) per tile keeps the kernel's
single-pointer contract (multi-element acquires neither link nor land
contiguously on this flow).
"""

import numpy as np
import torch
from ml_dtypes import bfloat16

ELEM = 13840
K_MAX = 6144


def quantize_and_pack(W, group_size=32, m_input=4, cols=8):
    """Quantize float tensor W (M, K) to signed w4 and pack universal tiles.

    Returns (packed uint8 buffer, W_dequant bf16 torch tensor).
    """
    M, K = W.shape
    assert K in (2048, 6144), "w4gemvu variants cover K in {2048, 6144}"
    assert K % group_size == 0
    assert M % cols == 0 and (M // cols) % m_input == 0

    num_groups_per_row = K // group_size
    tile_bytes = 8 + m_input * K // 2 + m_input * num_groups_per_row * 2
    slots = (tile_bytes + ELEM - 1) // ELEM
    assert slots == 1, "ELEM must cover the max-K tile in one slot"
    slot_bytes = ELEM

    Wg = torch.from_numpy(np.ascontiguousarray(W)).to(torch.float32)
    Wg = Wg.reshape(M * num_groups_per_row, group_size)
    amax = Wg.abs().amax(dim=1, keepdim=True)
    scale = (amax / 7.0).to(torch.bfloat16).to(torch.float32)
    q = torch.where(amax == 0, torch.zeros_like(Wg), torch.round(Wg / scale))
    q = torch.clamp(q, -8, 7).to(torch.int8)
    W_dequant = (q * scale).to(torch.bfloat16).reshape(M, K)

    tiles_per_col = M // cols // m_input
    packed = np.zeros(cols * tiles_per_col * slot_bytes, dtype=np.uint8)

    q_np = q.numpy().reshape(M, num_groups_per_row, group_size)
    scale_np = (
        scale.to(torch.bfloat16).view(torch.uint16).numpy().reshape(M, num_groups_per_row)
    )
    import struct

    k_le = struct.pack("<I", K)
    for col in range(cols):
        for t in range(tiles_per_col):
            row_start = col * (M // cols) + t * m_input
            off = (col * tiles_per_col + t) * slot_bytes
            # Layout v2: nibbles at 0 (row-major), 8-byte hole, scales,
            # K in the fixed slot tail. The anchors match what the
            # compiled kernel actually reads (peano drops the +8 on the
            # int4 weight stream; scales and the K load keep theirs).
            rows = q_np[row_start : row_start + m_input]
            lo = rows[:, :, 0::2].astype(np.uint8) & 0x0F
            hi = rows[:, :, 1::2].astype(np.uint8) & 0x0F
            nibbles = (lo | (hi << 4)).reshape(m_input, K // 2)
            packed[off : off + m_input * K // 2] = nibbles.reshape(-1)
            s = scale_np[row_start : row_start + m_input]
            o = off + m_input * K // 2 + 8
            packed[o : o + m_input * num_groups_per_row * 2] = s.reshape(-1).view(np.uint8)
            packed[off + ELEM - 8 : off + ELEM - 4] = np.frombuffer(k_le, dtype=np.uint8)

    return packed, W_dequant


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
