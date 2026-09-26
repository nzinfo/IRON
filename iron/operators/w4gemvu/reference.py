# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Reference for the universal (self-describing-block) w4gemvu operator.

Layout v5 = v4 matrix-unit tiles (P11) with the B STREAM REMOVED (P12:
dedicated B channels + F-slot fills cost ~30% of the weight stream; 8
channels already reach the ~55 GB/s device wall). The activation rides
the A fifo as a K=0 element:

  y[n] = sum_g sf[n][g] * d[g] * (int32 dot of W[n][g,*] x[g,*])

Weight quantization keeps the signed-int4 ABI (per-group-32 symmetric,
scale = amax/7 bf16); the activation is quantized int8 per-group-32
(scale = amax/127 bf16) so the inner product is an EXACT int32 dot.

Blocks are UNIFORM: one fifo element = 18560 B = ONE 16-row x 2048-k
tile. K=6144 ops stream 3 chunk-blocks per tile in CHUNK-MAJOR order
(the export layout, kept); blocks are self-describing via the chunk id
word, so the order is no longer load-bearing. Every kernel call
computes a 16-row partial; the host sums the 3 chunk partials for
K=6144. Each column's C carries one leading ZERO 16-row element from
the activation element (the host skips it).

Block layout (see w4gemvu.cc):
  [0 .. 16384)   nibbles, group-major: group g = 256 nibbles k-major,
                 element g*256 + k*16 + n (n = tile row) — the B-operand
                 order of mac_4x16_16x16 (a byte packs rows n, n+1).
  [16384 .. 18432) sf_t: bf16[64][16], row n's group-g scale at g*16+n
                 (transposed vs v3: one 32B load feeds the scale step).
  [18552 .. 18556) K as u32: 2048 = weight block, 0 = activation
                 element, else stale -> zero rows.
  [18556 .. 18560) chunk as u32: which 2048-wide x slice this block
                 consumes (0 for K=2048 ops).

Activation element (ELEM B): [0..6144) x int8 ALL chunks (chunk c at
c*2048), [6144..6528) d bf16[192] all chunks, [18552..18556) K = 0.
ONE per op, shared by every column's X fill — the per-token host cost
is a single ELEM write (no F-slot replication).

ABI: M % 256 == 0 for every K (16-row tiles x 8 cols; K=6144 needs
even tiles per chunk for the chunk-major sections — same bound).
"""

import numpy as np
import torch
from ml_dtypes import bfloat16

ELEM = 18560         # A fifo element = one 16-row x 2048 tile. %64 == 0 is
                     # MANDATORY on aie2p: depth-2 fifo element buffers sit
                     # at base and base+ELEM and a 1024-bit load_v stream
                     # (our int4 B operand) needs 64B alignment; %32-only
                     # left the nibbles garbage on every odd element buffer.
K_MAX = 6144
TILE_K = 2048        # k per block (K=6144 ops stream 3 blocks per tile)
TILE_ROWS = 16
D_BYTES_ALL = (K_MAX // 32) * 2  # all-chunk d bytes in the x element (384)


def chunks_per_tile(k):
    return k // TILE_K  # 1 (K=2048) or 3 (K=6144)


def tiles_per_chunk(M, cols=8):
    """16-row tiles per column per chunk (M is the PADDED row count)."""
    return M // cols // TILE_ROWS


def blocks_per_col(M, K, m_input=TILE_ROWS, cols=8):
    """Fifo blocks one column streams (chunk-major: chunk c's tiles
    consecutive — the export layout; blocks are self-describing via the
    chunk word so the order is conventional, not load-bearing)."""
    return tiles_per_chunk(M, cols) * chunks_per_tile(K)


def quantize_vector(x, group_size=32):
    """bf16 (K,) activation -> per-group-32 symmetric int8.

    Returns (q int8 (K,), d bf16 (K//32,), x_dequant f32 (K,)).
    The kernel's inner dot is EXACT int32 (|q_w*q_x|*32 <= 2^15); d and
    the weight sf are applied in f32 afterwards.
    """
    K = x.numel()
    assert K % group_size == 0
    xg = x.to(torch.float32).reshape(-1, group_size)
    amax = xg.abs().amax(dim=1, keepdim=True)
    d = (amax / 127.0).to(torch.bfloat16)
    df = d.to(torch.float32)
    q = torch.where(amax == 0, torch.zeros_like(xg), torch.round(xg / df))
    q = torch.clamp(q, -127, 127).to(torch.int8)
    x_deq = (q.to(torch.float32) * df).reshape(K)
    return q.reshape(K), d.reshape(-1), x_deq


def quantize_and_pack(W, group_size=32, m_input=TILE_ROWS, cols=8):
    """Quantize float tensor W (M, K) to signed w4 and pack v4 tiles.

    M must satisfy the v4 ABI: M % 256 == 0 (16-row tiles / 8 cols with
    blocks paired two per B slot). Pad with zero rows upstream.

    Returns (packed uint8 buffer, W_dequant bf16 torch tensor).
    """
    M, K = W.shape
    assert K in (2048, 6144), "w4gemvu variants cover K in {2048, 6144}"
    assert K % group_size == 0
    assert M % cols == 0 and (M // cols) % m_input == 0
    chunks = chunks_per_tile(K)
    T = tiles_per_chunk(M, cols)

    num_groups_per_row = K // group_size
    groups_per_chunk = TILE_K // group_size  # 64

    Wg = torch.from_numpy(np.ascontiguousarray(W)).to(torch.float32)
    Wg = Wg.reshape(M * num_groups_per_row, group_size)
    amax = Wg.abs().amax(dim=1, keepdim=True)
    scale = (amax / 7.0).to(torch.bfloat16).to(torch.float32)
    q = torch.where(amax == 0, torch.zeros_like(Wg), torch.round(Wg / scale))
    q = torch.clamp(q, -8, 7).to(torch.int8)
    W_dequant = (q * scale).to(torch.bfloat16).reshape(M, K)

    rows_per_col = M // cols
    blocks = T * chunks
    packed = np.zeros(cols * blocks * ELEM, dtype=np.uint8)

    q_np = q.numpy().reshape(M, num_groups_per_row, group_size)
    scale_np = (
        scale.to(torch.bfloat16).view(torch.uint16).numpy().reshape(M, num_groups_per_row)
    )
    import struct

    k_le = struct.pack("<I", TILE_K)  # the TILE K — always 2048
    for col in range(cols):
        base_rows = col * rows_per_col
        for c in range(chunks):
            c_le = struct.pack("<I", c)
            for t in range(T):
                off = (col * blocks + c * T + t) * ELEM
                packed[off + ELEM - 8 : off + ELEM - 4] = np.frombuffer(k_le, dtype=np.uint8)
                packed[off + ELEM - 4 : off + ELEM] = np.frombuffer(c_le, dtype=np.uint8)
                r0 = base_rows + t * m_input
                # Nibbles: group g's 256 elements k-major [k][n] — byte
                # j of the group packs rows 2j (low) and 2j+1 (high) at
                # k = j // 8. This is the B-operand order the matrix
                # unit consumes (mac_4x16_16x16 B is [16k x 16n]).
                nib = (
                    q_np[r0 : r0 + m_input, c * groups_per_chunk : (c + 1) * groups_per_chunk, :]
                    .transpose(1, 2, 0)
                    .astype(np.uint8)
                    & 0x0F
                )  # (64, 32, 16)
                pairs = nib.reshape(groups_per_chunk, group_size, m_input // 2, 2)
                bts = (pairs[..., 0] | (pairs[..., 1] << 4)).reshape(-1)
                packed[off : off + m_input * TILE_K // 2] = bts
                # Transposed scales: row n's group-g scale at g*16 + n.
                s = scale_np[r0 : r0 + m_input, c * groups_per_chunk : (c + 1) * groups_per_chunk]
                packed[off + m_input * TILE_K // 2 : off + m_input * TILE_K // 2 + m_input * groups_per_chunk * 2] = (
                    s.transpose(1, 0).reshape(-1).view(np.uint8)
                )

    return packed, W_dequant


def c_section_rows(M, K, m_input=TILE_ROWS, cols=8):
    """bf16 ROWS per column's C section: one zero element (the
    activation's) + the block partials + one pad element. Sections are
    sized in whole 16-row elements and must be an EVEN element count
    (BD transfers and offsets are 4-byte aligned and 1+blocks is always
    odd); the pad element is never transferred or read."""
    return (blocks_per_col(M, K, m_input, cols) + 2) * m_input


def output_rows(M, K):
    """bf16 rows the C drain tensor holds: per column one zero 16-row
    element from the activation, the block partials, and one pad
    element."""
    return 8 * c_section_rows(M, K)  # cols is always 8 in the universal design


def unshuffle_output(c_rows, M, K, cols=8):
    """Device C rows -> real (M,) output. v5: each column's section
    starts with the activation element's 16 zero rows (skip) and ends
    with a pad element, holding its chunk-major partials in between
    (K=6144 sums the 3 chunk partials)."""
    chunks = chunks_per_tile(K)
    rows_per_col = M // cols
    section = c_section_rows(M, K)
    raw = c_rows.reshape(cols, section)
    live = raw[:, TILE_ROWS : TILE_ROWS + chunks * rows_per_col]
    if chunks == 1:
        return live.reshape(M).clone()
    return live.reshape(cols, chunks, rows_per_col).sum(dim=1).reshape(M)


def shuffle_output(partials, M, K, cols=8):
    """Per-chunk (chunks, M) bf16 partials -> the device C-row layout:
    column col's section = [16 zero rows (activation) | chunk-major
    partial sections | pad element]."""
    chunks = chunks_per_tile(K)
    assert partials.shape == (chunks, M)
    rows_per_col = M // cols
    section = c_section_rows(M, K)
    raw = torch.zeros(cols * section, dtype=partials.dtype)
    for col in range(cols):
        base = col * section + TILE_ROWS  # skip the x zeros
        for c in range(chunks):
            raw[base + c * rows_per_col : base + (c + 1) * rows_per_col] = \
                partials[c, col * rows_per_col : (col + 1) * rows_per_col]
    return raw


def build_activation_element(q, d, K):
    """(q int8 (K,), d bf16 (K//32,)) -> the ONE ELEM-sized activation
    element all columns share: x all chunks at 0..K, d all chunks at
    K_MAX, K header 0 at ELEM-8."""
    import struct

    xb = np.zeros(ELEM, dtype=np.uint8)
    xb[0:K] = q.numpy().view(np.uint8)
    xb[K_MAX : K_MAX + (K // 32) * 2] = d.view(torch.uint16).numpy().view(np.uint8)
    xb[ELEM - 8 : ELEM - 4] = np.frombuffer(struct.pack("<I", 0), dtype=np.uint8)
    return torch.from_numpy(xb)


def generate_golden_reference(M, K, group_size=32, m_input=TILE_ROWS, cols=8, seed=42):
    """Random signed weights + deterministic x, packed and golden.

    The golden output carries BOTH the per-chunk partials (the raw
    device C layout, chunk-major M-row sections) and the final (M,)
    sum — K=6144 rounds each partial to bf16 on device and the host
    sums the three.
    """
    torch.manual_seed(seed)
    W = (torch.rand(M, K, dtype=torch.float32) * 2 - 1).numpy()
    x = (torch.rand(K, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)

    packed, W_dequant = quantize_and_pack(W, group_size, m_input, cols)
    q, d, x_deq = quantize_vector(x, group_size)

    chunks = chunks_per_tile(K)
    partials = torch.empty(chunks, M, dtype=torch.bfloat16)
    Wf = W_dequant.to(torch.float32)
    for c in range(chunks):
        k0 = c * TILE_K
        partials[c] = (Wf[:, k0 : k0 + TILE_K] @ x_deq[k0 : k0 + TILE_K]).to(torch.bfloat16)
    out = partials[0] if chunks == 1 else (
        partials.sum(dim=0).to(torch.bfloat16)
    )

    return {
        "packed_weights": packed,
        "x": x,
        "activation": build_activation_element(q, d, K),
        "output": out,
        "output_raw": shuffle_output(partials, M, K),
    }
