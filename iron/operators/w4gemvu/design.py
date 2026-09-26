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
UNIVERSAL fused INT4-dequant GEMV (M3b), layout v5 = B STREAM REMOVED
(P12): the P12 probe proved 8 fill channels already reach the ~55 GB/s
device wall once the B stream is gone (v4's dedicated B channels + F-slot
fills + per-slot lock traffic cost ~30% of lm_head), and that doubling A
to 16 channels adds NOTHING. v5 keeps ONE fifo per column and rides the
activation on it:

  - A fifo element = ELEM = 18560 bytes, ALWAYS. Two flavors, told apart
    by the K header at ELEM-8 (the kernel's guard):
      * activation element (K=0): x int8 all chunks + d bf16[192], one
        per op, filled FIRST from the X tensor — fifo order IS the
        per-op barrier (a parked core cannot reach an op's blocks before
        consuming that op's activation; the v4 in-loop B acquire played
        exactly this role). Emits one zero C element the host skips.
      * weight block (K=2048, chunk id at ELEM-4): ONE 16x2048 tile, ONE
        kernel call each; blocks are self-describing so the stream order
        is free (K=6144 ops keep v4's per-column chunk-major order only
        because the export layout already is).
  - C fifo element = m_input bf16 = 32 B; depth 2. Every C row is live
    (K=6144 rows are chunk partials the host sums) except each column's
    leading x element.

Fixed device geometry (must match across variants byte-for-byte so the
compiled PDI is bit-identical): 8 columns, m_input=16, ELEM % 64 == 0
(element buffers at base/base+ELEM; aie2p load_v needs 64B alignment for
1024-bit vectors). acquire(1) per element keeps the kernel's
single-pointer contract (multi-element acquires are NOT an option: the
placer does not allocate a fifo's elements contiguously, and the
dynamic-objFifo lowering of acquire(n>1) fails to link with `undefined
symbol: A_L3L1_0_cons_buff_1` on this flow). Depth 2 = double-buffered.

Variant parameters (ctrl-code only): M (padded rows) and K (2048 or
6144) -> DDR block layout and tap sizes. ABI: M % 256 == 0 for every K.
"""

ELEM = 18560         # A fifo element = one 16-row x 2048 tile (%64 == 0:
                     # element buffers sit at base/base+ELEM; aie2p load_v
                     # needs 64B alignment for 1024-bit vectors)
K_MAX = 6144
M_INPUT = 16
TILE_K = 2048


def chunks_per_tile(k):
    return k // TILE_K  # 1 (K=2048) or 3 (K=6144)


def my_w4gemvu(dev, cols, M, K, group_size=32):
    assert cols == 8, "universal design is laid out for 8 columns"
    assert K in (2048, 6144), "K must be 2048 or 6144"
    assert group_size == 32
    chunks = chunks_per_tile(K)
    blocks_per_col = (M // cols // M_INPUT) * chunks

    dtype_in = np.dtype[np.uint8]
    dtype_out = np.dtype[bfloat16]

    dev_ty = NPU1() if dev == "npu" else NPU2()

    # Per-column DDR sizes (uniform blocks stream only live bytes). The
    # weights tensor is PURE blocks — no B slots (v4's A tap had to stride
    # over them; v5 fills are single linear BDs, the fastest form).
    bytes_per_col = blocks_per_col * ELEM
    packed_total_bytes = cols * bytes_per_col

    # One extra zero C element per column (the activation element's),
    # plus one 16-row PAD element per column: sections are 1+blocks
    # ELEMENTS (always ODD — blocks is even), and BD offsets/transfers
    # need 4-byte alignment, so a column's section must start and size
    # to even elements. The pad element is never transferred. c_rows is
    # in bf16 SCALARS (the tap/tensor unit — v4's C tensor was M rows).
    c_rows = cols * (blocks_per_col + 2) * M_INPUT

    # L1 types — the fixed geometry shared by every variant.
    L1_A_ty = np.ndarray[(ELEM,), dtype_in]
    L1_C_ty = np.ndarray[(M_INPUT,), dtype_out]

    # L3 (DDR) types
    L3_A_ty = np.ndarray[(packed_total_bytes,), dtype_in]
    L3_X_ty = np.ndarray[(ELEM,), dtype_in]  # ONE activation element,
    # shared by every column's X fill (all taps read offset 0 — the v4
    # B-tap pattern of one common source, now 1 element instead of F
    # slots, so the per-token host cost is a single 18.5 KB write).
    L3_C_ty = np.ndarray[(c_rows,), dtype_out]

    fused_matvec = Kernel(
        "w4gemvu_matvec_bf16",
        "w4gemvu.o",
        [
            np.int32,   # m (rows per tile, compiled-in: M_INPUT)
            L1_A_ty,    # self-describing block (K=0 -> activation staging)
            L1_C_ty,
            np.int32,   # group_size
            np.int32,   # tile_idx (vestigial v3 ABI slot; always 0)
        ],
    )

    A_L3L1_fifos = [
        ObjectFifo(L1_A_ty, name=f"A_L3L1_{i}", depth=2)
        for i in range(cols)
    ]
    C_L1L3_fifos = [
        ObjectFifo(L1_C_ty, name=f"C_L1L3_{i}", depth=2) for i in range(cols)
    ]

    def core_body(A_L3L1_fifo, C_L1L3_fifo, fused_matvec_fn):
        # Uniform per-element loop — the kernel's K-header guard sorts
        # activation vs weight elements, and the fill order (X element
        # first, per column) makes fifo order the per-op barrier. All
        # loop bounds are compile-time so the device side stays identical
        # across variants; the ctrl code decides the run length via the
        # number of A elements.
        for _ in range_(0xFFFFFFFF):
            a = A_L3L1_fifo.acquire(1)
            c = C_L1L3_fifo.acquire(1)
            fused_matvec_fn(M_INPUT, a, c, 32, 0)
            C_L1L3_fifo.release(1)
            A_L3L1_fifo.release(1)

    workers = [
        Worker(
            core_body,
            [
                A_L3L1_fifos[i].cons(),
                C_L1L3_fifos[i].prod(),
                fused_matvec,
            ],
        )
        for i in range(cols)
    ]

    # The activation element fill MUST be issued before the column's
    # weight fill (same fifo/channel; ctrl ops execute in order).
    X_taps = [
        TensorAccessPattern(
            tensor_dims=(1, ELEM),
            offset=0,
            sizes=[1, 1, 1, ELEM],
            strides=[0, 0, 0, 1],
        )
        for _ in range(cols)
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
    # C taps as 2D (elements x rows): a per-column transfer is 1+blocks
    # ELEMENTS — ODD, so the collapsed 1D form is 17x2=34 B and trips
    # the BD "multiple of 4 bytes"/offset-alignment rules. Element/row
    # dims keep every BD transfer a whole number of 32-B elements at
    # 4B-aligned section offsets (blocks even => section = blocks+2
    # elements = even*32 B). Offsets/strides are in bf16 scalars.
    C_taps = [
        TensorAccessPattern(
            tensor_dims=(1, c_rows),
            offset=col * (blocks_per_col + 2) * M_INPUT,
            sizes=[1, 1, 1 + blocks_per_col, M_INPUT],
            strides=[0, 0, M_INPUT, 1],
        )
        for col in range(cols)
    ]

    rt = Runtime()
    with rt.sequence(L3_A_ty, L3_X_ty, L3_C_ty) as (A, X, C):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(cols):
            # One-element activation fill (K=0), then the column's pure
            # block stream — both linear BDs.
            rt.fill(A_L3L1_fifos[i].prod(), X, X_taps[i], task_group=tg)
            rt.fill(A_L3L1_fifos[i].prod(), A, A_taps[i], task_group=tg)
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
