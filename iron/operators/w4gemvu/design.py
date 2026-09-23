# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
UNIVERSAL fused INT4-dequant GEMV (M3b): one PDI for every projection
shape. The kernel reads K from a self-describing slot tail at runtime,
so the device side (workers, fifos, placement) is IDENTICAL for all
shapes and only the ctrl-code differs per (M, K) — run-w4layer showed
every CU switch costs ~650us of PDI reload; with all ops on one CU the
whole decode projection stream runs at single-CU speed (13+ tok/s).

Fixed device geometry (must match across variants byte-for-byte so the
compiled PDI is bit-identical):
  - 8 columns, m_input=4 rows per tile/kernel call
  - A fifo element = ELEM = 13840 bytes = ONE padded max-K tile: a
    K=6144 tile is 13832 B (8 B slack), a K=2048 tile is 4616 B (9224 B
    of DDR padding). Every tile is exactly acquire(1): the kernel keeps
    its single-pointer contract. (Multi-element acquires are NOT an
    option here: the placer does not allocate a fifo's elements
    contiguously — observed 0x44000/0x48000/0x4C000/0x45210 — and the
    dynamic-objFifo lowering of acquire(n>1) fails to link with
    `undefined symbol: A_L3L1_0_cons_buff_1` on this flow.) Depth 2 =
    double-buffered tiles; the K=2048 fill moves ~3x the bytes but the
    AIE reads only the live prefix, so its consumption rate is
    unchanged and the padding lands on the shim DMA engines.
  - B fifo element = K_MAX*2 = 12288 B (K=2048 ops zero-pad the slot);
    depth 2 (ping-pong, same shape as the A path). The activation is
    streamed as ONE multi-element fill per column: the DDR vector buffer
    holds F = tiles_per_col/TILES_PER_B copies of x back to back (host
    replicates it), so the fill is a single large BD with a positive
    stride — untapped per-element fills would need F shim BDs and the
    allocator caps a channel at 16 (gate_up F=24 exhausts it).
  - C fifo element = m_input bf16 = 8 B; depth 2.

Variant parameters (ctrl-code only): M (rows) and K (2048 or 6144) ->
DDR tile-slot layout (every tile padded to ELEM) and tap offsets.
"""

ELEM = 13840         # A fifo element bytes = one padded max-K tile
ELEMS_PER_TILE = 1   # acquire(1): single-pointer kernel contract
K_MAX = 6144
M_INPUT = 4
TILES_PER_B = 16     # tiles served per B acquire (every shape divides)


def tile_slots(k):
    """Fifo elements one padded tile occupies."""
    tile_bytes = 8 + M_INPUT * k // 2 + M_INPUT * (k // 32) * 2
    return (tile_bytes + ELEM - 1) // ELEM


def my_w4gemvu(dev, cols, M, K, group_size=32):
    assert cols == 8, "universal design is laid out for 8 columns"
    assert K in (2048, 6144), "K must be 2048 or 6144"
    assert group_size == 32
    slots = tile_slots(K)
    assert slots <= ELEMS_PER_TILE
    assert (M // cols // M_INPUT) % TILES_PER_B == 0, \
        "tiles per column must divide TILES_PER_B for the B reuse loop"

    dtype_in = np.dtype[np.uint8]
    dtype_vec = np.dtype[bfloat16]
    dtype_out = np.dtype[bfloat16]

    dev_ty = NPU1() if dev == "npu" else NPU2()

    # Per-column DDR sizes (padded tile slots keep fills element-aligned).
    rows_per_col = M // cols
    tiles_per_col = rows_per_col // M_INPUT
    bytes_per_col = tiles_per_col * slots * ELEM
    packed_total_bytes = cols * bytes_per_col

    # L1 types — the fixed geometry shared by every variant.
    L1_A_ty = np.ndarray[(ELEM,), dtype_in]
    L1_B_ty = np.ndarray[(K_MAX,), dtype_vec]
    L1_C_ty = np.ndarray[(M_INPUT,), dtype_out]

    # L3 (DDR) types
    b_elems = (tiles_per_col // TILES_PER_B) * K_MAX
    L3_A_ty = np.ndarray[(packed_total_bytes,), dtype_in]
    L3_B_ty = np.ndarray[(b_elems,), dtype_vec]
    L3_C_ty = np.ndarray[(M,), dtype_out]

    fused_matvec = Kernel(
        "w4gemvu_matvec_bf16",
        "w4gemvu.o",
        [
            np.int32,   # m (rows per tile, compiled-in: M_INPUT)
            L1_A_ty,    # self-describing tile
            L1_B_ty,    # activation (first K*2 bytes valid)
            L1_C_ty,
            np.int32,   # group_size
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
        # w4gemv2's proven shape: b acquired INSIDE the outer loop, one B
        # slot per TILES_PER_B tiles. The in-loop acquire is also the
        # per-RUN synchronization — a core parked from a previous run
        # (with a different activation) can never compute against a stale
        # or mid-fill B slot (hoisting b out of the loop races exactly
        # that way: the first tiles read the old slot while the fill's BD
        # overwrites it). Tiles per pass are compile-time so the device
        # side stays identical across variants; the ctrl-code decides the
        # pass count via the number of B fills.
        for _ in range_(0xFFFFFFFF):
            b = B_L3L1_fifo.acquire(1)
            for _t in range_(TILES_PER_B):
                a = A_L3L1_fifo.acquire(ELEMS_PER_TILE)
                c = C_L1L3_fifo.acquire(1)
                fused_matvec_fn(M_INPUT, a, b, c, 32)
                A_L3L1_fifo.release(ELEMS_PER_TILE)
                C_L1L3_fifo.release(1)
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
            tensor_dims=(1, M),
            offset=col * rows_per_col,
            sizes=[1, 1, 1, rows_per_col],
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
            # One multi-element B fill per column: the vector buffer holds
            # F copies of the activation, so the stream shape mirrors the
            # proven A fill (one big BD, positive stride). Untapped
            # per-element fills would need F shim BDs per channel and the
            # allocator caps a channel at 16 (gate_up F=24 exhausts it);
            # a stride-0 repeat tap is rejected by NPU BDs.
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
