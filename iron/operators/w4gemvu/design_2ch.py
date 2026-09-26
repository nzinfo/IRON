# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""P12 probe: weight-stream channel scaling (B stream removed).

The v4 placement puts A on shims 0-3 (8 MM2S channels) and B on shims 4-7
(8 channels) — B burns half the chip's fill bandwidth for ~1% of the
bytes. Measured per-channel rate ~4.9 GB/s (lm_head 38.9 GB/s / 8 ch), so
if the rate is channel-limited, A on 16 channels should be ~2x.

L1 geometry: two full 18560-B element fifos per core (4 x 18560 = 74 KB)
do NOT fit the 64-KB tile data memory. The probe instead halves the
element (9280 B, still %64==0): each 18560-B block streams as two halves.
The kernel is UNCHANGED: its K-header guard (w4gemvu.cc, header at
+18552) reads into the adjacent L1 buffer — in bounds, garbage value
!= 2048 — so every call takes the zero-rows path. Byte count on the wire
is IDENTICAL to production; the core-side work is a few instructions per
element (compute was ~0.3us per block in v4 anyway). PERF ONLY — outputs
are garbage.

channels=1 vs 2 isolates the two effects the first probe mixed: with
channels=1 the full column streams over ONE fifo (8 columns = 8 MM2S
channels, same count as production) but B is still gone — so any gain
over production is B-removal/fill-count, and the 1ch->2ch delta is the
true channel scaling. Each channel's range is a contiguous LINEAR slice
of the column (auto-split into chained BDs — multi-dim taps hit the
4-dim BD limit and the 10-bit per-dim size limit).
"""

import numpy as np
from ml_dtypes import bfloat16
import argparse

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2

ELEM = 18560      # one full 16-row x 2048 tile (production block size)
ELEM_HALF = 9280  # probe fifo element: half a block, %64 == 0
K_MAX = 6144
M_INPUT = 16
TILE_K = 2048


def my_w4gemvu_2ch(dev, cols, M, K, group_size=32, channels=2):
    assert cols == 8
    assert K in (2048, 6144)
    assert group_size == 32
    assert channels in (1, 2)
    chunks = K // TILE_K
    blocks_per_col = (M // cols // M_INPUT) * chunks

    dtype_in = np.dtype[np.uint8]
    dtype_out = np.dtype[bfloat16]

    dev_ty = NPU1() if dev == "npu" else NPU2()

    bytes_per_col = blocks_per_col * ELEM
    packed_total_bytes = cols * bytes_per_col
    # Channel c streams the c-th contiguous slice of each column's bytes.
    # The core runs ONE iteration (one C element) per A element per fifo —
    # the C count MUST equal elements-per-fifo or the core over-produces,
    # blocks on the depth-2 C fifo, the fills stall and the task group
    # deadlocks (the channels=1 bug: 2*blocks elements vs blocks C).
    seg_bytes = bytes_per_col // channels
    assert seg_bytes % ELEM_HALF == 0  # slices tile whole 9280-B elements
    elements_per_fifo = seg_bytes // ELEM_HALF
    c_rows = cols * elements_per_fifo * M_INPUT

    L1_A_ty = np.ndarray[(ELEM_HALF,), dtype_in]
    L1_C_ty = np.ndarray[(M_INPUT,), dtype_out]

    L3_A_ty = np.ndarray[(packed_total_bytes,), dtype_in]
    L3_C_ty = np.ndarray[(c_rows,), dtype_out]

    fused_matvec = Kernel(
        "w4gemvu_matvec_bf16",
        "w4gemvu.o",
        [
            np.int32,        # m
            L1_A_ty,         # block (probe: half element, garbage K -> zero rows)
            L1_C_ty,
            np.int32,        # group_size
            np.int32,        # tile_idx (vestigial)
        ],
    )

    # A_fifos[col][channel]
    A_fifos = [
        [
            ObjectFifo(L1_A_ty, name=f"A{c}_L3L1_{i}", depth=2)
            for c in range(channels)
        ]
        for i in range(cols)
    ]
    C_fifos = [
        ObjectFifo(L1_C_ty, name=f"C_L1L3_{i}", depth=2) for i in range(cols)
    ]

    def core_body(*args):
        A_fifo_list = args[:-2]
        C_fifo, fn = args[-2], args[-1]
        for _ in range_(0xFFFFFFFF):
            for _j in range_(1):  # one block (`channels` half-elements) per iter
                acq = [f.acquire(1) for f in A_fifo_list]
                c0 = C_fifo.acquire(1)
                fn(M_INPUT, acq[0], c0, 32, 0)
                C_fifo.release(1)
                for f in A_fifo_list:
                    f.release(1)

    workers = [
        Worker(
            core_body,
            [
                *[A_fifos[i][c].cons() for c in range(channels)],
                C_fifos[i].prod(),
                fused_matvec,
            ],
        )
        for i in range(cols)
    ]

    # Channel c streams the c-th contiguous slice of each column's bytes:
    # linear ranges (the production fill pattern), so the lowering
    # auto-splits them into chained BDs. Same total bytes as production.
    A_taps = [
        [
            TensorAccessPattern(
                tensor_dims=(1, packed_total_bytes),
                offset=col * bytes_per_col + c * seg_bytes,
                sizes=[1, 1, 1, seg_bytes],
                strides=[0, 0, 0, 1],
            )
            for col in range(cols)
        ]
        for c in range(channels)
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

    rt = Runtime()
    with rt.sequence(L3_A_ty, L3_C_ty) as (A, C):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(cols):
            for c in range(channels):
                rt.fill(A_fifos[i][c].prod(), A, A_taps[c][i], task_group=tg)
        for i in range(cols):
            rt.drain(C_fifos[i].cons(), C, C_taps[i], task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(prog="P12 channel probe")
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M", type=int, required=True)
    argparser.add_argument("-K", type=int, required=True)
    argparser.add_argument("--cols", type=int, default=8)
    argparser.add_argument("--channels", type=int, default=2)
    argparser.add_argument("--output-file-path", "-o", type=str, required=True)
    args = argparser.parse_args()
    module = my_w4gemvu_2ch(args.dev, args.cols, args.M, args.K, channels=args.channels)
    with open(args.output_file_path, "w") as f:
        f.write(str(module))
