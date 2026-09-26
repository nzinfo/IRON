# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
from pathlib import Path
from ml_dtypes import bfloat16
import argparse

import aie.dialects.index as index
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2

"""
UNIVERSAL fused INT4-dequant GEMV (M3b), layout v4 = MATRIX-UNIT TILES
(P11): ONE PDI for every projection shape, and the inner loop on the
aie2p matrix unit (mac_4x16_16x16 = 1024 MACs/instr; the v3 fp loop was
ISSUE-bound at ~10 vector ops per 32 MACs — the P11 probe measured the
matrix unit at ~100 GMAC/s/core, 32x). Every block is UNIFORM: 16 rows
x 2048 k, ONE kernel call each; K=6144 ops stream 3 chunk-blocks per
tile in chunk-major order (both blocks a B slot serves share one x
chunk), so the device side is shape-free and only ctrl code differs.

Fixed device geometry (must match across variants byte-for-byte so the
compiled PDI is bit-identical):
  - 8 columns, m_input=16 rows per tile/kernel call
  - A fifo element = ELEM = 18560 bytes = ONE 16x2048 tile (64-byte
    aligned nibbles, transposed scales, K header at ELEM-8). acquire(1)
    per block keeps the kernel's single-pointer contract (multi-element
    acquires are NOT an option: the placer does not allocate a fifo's
    elements contiguously, and the dynamic-objFifo lowering of
    acquire(n>1) fails to link with `undefined symbol:
    A_L3L1_0_cons_buff_1` on this flow). Depth 2 = double-buffered.
  - B fifo element = B_SLOT = 6528 bytes uint8: x int8 (K_MAX wide,
    live chunk at 0..2048) + per-group bf16 x scales d at K_MAX. One B
    slot serves BLOCKS_PER_B = 2 blocks, so the DDR vector buffer holds
    F = blocks/2 slots (host packs the chunk per slot) — a single large
    BD with a positive stride (per-element fills would need F shim BDs
    and the allocator caps a channel at 16).
  - C fifo element = m_input bf16 = 32 B; depth 2. Every C row is live
    (K=6144 rows are chunk partials the host sums — no v3 zero rows).

Variant parameters (ctrl-code only): M (padded rows) and K (2048 or
6144) -> DDR block layout and tap sizes. ABI: M % 256 == 0 for every K
(16-row tiles / 8 cols, blocks paired two per B slot, and K=6144 needs
even tiles per chunk — the same bound).
"""

ELEM = 18560         # A fifo element = one 16-row x 2048 tile (%64 == 0:
                     # element buffers sit at base/base+ELEM; aie2p load_v
                     # needs 64B alignment for 1024-bit vectors)
K_MAX = 6144
M_INPUT = 16
TILE_K = 2048
B_SLOT = K_MAX + 192 * 2  # 6528: x int8 (K_MAX) + d bf16[192]
BLOCKS_PER_B = 2     # blocks served per B slot (keeps fills single-BD)


def chunks_per_tile(k):
    return k // TILE_K  # 1 (K=2048) or 3 (K=6144)


def my_w4gemvu(dev, cols, M, K, group_size=32):
    assert cols == 8, "universal design is laid out for 8 columns"
    assert K in (2048, 6144), "K must be 2048 or 6144"
    assert group_size == 32
    chunks = chunks_per_tile(K)
    blocks_per_col = (M // cols // M_INPUT) * chunks
    assert blocks_per_col % BLOCKS_PER_B == 0, \
        "blocks per column must be even (B slot serves two blocks)"

    dtype_in = np.dtype[np.uint8]
    dtype_vec = np.dtype[np.uint8]
    dtype_out = np.dtype[bfloat16]

    dev_ty = NPU1() if dev == "npu" else NPU2()

    # Per-column DDR sizes (uniform blocks stream only live bytes).
    bytes_per_col = blocks_per_col * ELEM
    packed_total_bytes = cols * bytes_per_col

    # Every block emits one live 16-row element; K=6144 rows are chunk
    # partials in chunk-major M-row sections (host sums them).
    c_rows = M * chunks

    # L1 types — the fixed geometry shared by every variant.
    L1_A_ty = np.ndarray[(ELEM,), dtype_in]
    L1_B_ty = np.ndarray[(B_SLOT,), dtype_vec]
    L1_C_ty = np.ndarray[(M_INPUT,), dtype_out]

    # L3 (DDR) types
    b_elems = (blocks_per_col // BLOCKS_PER_B) * B_SLOT
    L3_A_ty = np.ndarray[(packed_total_bytes,), dtype_in]
    L3_B_ty = np.ndarray[(b_elems,), dtype_vec]
    L3_C_ty = np.ndarray[(c_rows,), dtype_out]

    fused_matvec = Kernel(
        "w4gemvu_matvec_bf16",
        "w4gemvu.o",
        [
            np.int32,   # m (rows per tile, compiled-in: M_INPUT)
            L1_A_ty,    # self-describing block
            L1_B_ty,    # x int8 + d scales (chunk at 0..2048, d at K_MAX)
            L1_C_ty,
            np.int32,   # group_size
            np.int32,   # tile_idx (vestigial v3 ABI slot; always 0)
        ],
    )

    A_L3L1_fifos = [
        ObjectFifo(L1_A_ty, name=f"A_L3L1_{i}", depth=2)
        for i in range(cols)
    ]
    B_L3L1_fifos = [
        ObjectFifo(L1_B_ty, name=f"B_L3L1_{i}", depth=2) for i in range(cols)
    ]
    C_L1L3_fifos = [
        ObjectFifo(L1_C_ty, name=f"C_L1L3_{i}", depth=2) for i in range(cols)
    ]

    def core_body(A_L3L1_fifo, B_L3L1_fifo, C_L1L3_fifo, fused_matvec_fn):
        # The in-loop B acquire is also the per-RUN synchronization — a
        # core parked from a previous run (with a different activation)
        # can never compute against a stale or mid-fill B slot (hoisting
        # b out of the loop races exactly that way: the first tiles read
        # the old slot while the fill's BD overwrites it). All loop
        # bounds are compile-time so the device side stays identical
        # across variants; the ctrl-code decides the run length via the
        # number of A blocks.
        for _ in range_(0xFFFFFFFF):
            b = B_L3L1_fifo.acquire(1)
            for _blk in range_(BLOCKS_PER_B):
                a = A_L3L1_fifo.acquire(1)
                c = C_L1L3_fifo.acquire(1)
                fused_matvec_fn(M_INPUT, a, b, c, 32, 0)
                C_L1L3_fifo.release(1)
                A_L3L1_fifo.release(1)
            B_L3L1_fifo.release(1)

    workers = [
        Worker(
            core_body,
            [
                A_L3L1_fifos[i].cons(),
                B_L3L1_fifos[i].cons(),
                C_L1L3_fifos[i].prod(),
                fused_matvec,
            ],
        )
        for i in range(cols)
    ]

    A_taps = [
        TensorAccessPattern(
            tensor_dims=(1, packed_total_bytes),
            offset=col * bytes_per_col,
            sizes=[1, 1, 1, bytes_per_col],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]
    C_taps = [
        TensorAccessPattern(
            tensor_dims=(1, c_rows),
            offset=col * (c_rows // cols),
            sizes=[1, 1, 1, c_rows // cols],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]
    B_taps = [
        TensorAccessPattern(
            tensor_dims=(1, b_elems),
            offset=0,
            sizes=[1, 1, 1, b_elems],
            strides=[0, 0, 0, 1],
        )
        for _ in range(cols)
    ]

    rt = Runtime()
    with rt.sequence(L3_A_ty, L3_B_ty, L3_C_ty) as (A, B, C):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(cols):
            rt.fill(A_L3L1_fifos[i].prod(), A, A_taps[i], task_group=tg)
            # One multi-element B fill per column: the vector buffer
            # holds F slots of the packed (x, d) activation, so the
            # stream shape mirrors the proven A fill (one big BD,
            # positive stride). Untapped per-element fills would need F
            # shim BDs per channel and the allocator caps a channel at
            # 16; a stride-0 repeat tap is rejected by NPU BDs.
            rt.fill(B_L3L1_fifos[i].prod(), B, B_taps[i], task_group=tg)
        for i in range(cols):
            rt.drain(
                C_L1L3_fifos[i].cons(),
                C,
                C_taps[i],
                task_group=tg,
                wait=True,
            )
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog="AIE universal fused dequant GEMV MLIR design",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M", type=int, required=True)
    argparser.add_argument("-K", type=int, required=True)
    argparser.add_argument("--cols", type=int, default=8)
    argparser.add_argument("--output-file-path", "-o", type=str, required=True)
    args = argparser.parse_args()
    module = my_w4gemvu(args.dev, args.cols, args.M, args.K)
    with open(args.output_file_path, "w") as f:
        f.write(str(module))
