# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
from pathlib import Path
import argparse

from aie.dialects.aiex import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2

"""M6 FUSED rms-pair GEMV (P16): op1 -> add+rms+quantize -> op2 in ONE
exec. The layer's op stream is CPU-interlocked (every op's x needs the
previous op's C on the host), which forced one exec per op and the
~100 us machinery floor x 129 ops/token (T_min probe, notes perf-lab
P16). This design removes ONE exec + ONE host round trip per pair by
expressing the glue ON DEVICE:

  tg1 (task group 1) — EXACTLY the v5 single-op shape:
    per column: X fill (op1's K=0 activation element), A1 fill (op1's
    blocks) — then C drains of op1's sections, wait=True.
  tg2 — also exactly the v5 shape (2 fills/fifo + drains):
    per column: rms fill — ONE element sourced from the C TENSOR itself
    (rows [0..9280): op1's sections + the residual the host pre-wrote
    into the window's free rows + the K=1/blocks1 header words), then
    the A2 fill — [w/glue element (K=3) | op2's blocks] contiguous, so
    the glue element streams FIRST and fifo order runs it before any
    op2 block (the v5 in-fifo barrier). Then C drains of op2's sections,
    wait=True.

BOTH task groups carry 4 fills + 2 drains per shim — the ctrl
generator pins S2MM drain queue-values to slots 4/5, so any group with
MORE than 4 preceding fills on a shim desyncs the drain's value from
its descriptor address and deadlocks the runlist (first fused attempt
hung exactly there; see notes perf-lab P16).

Kernel flavors (w4gemvu.cc, self-describing K header):
  K=0     op1's activation (host-quantized as today) -> x_stage prebuild
  K=2048  weight block (unchanged hot loop)
  K=1     the C window: compact-stage op1's partials + residual
  K=3     the w element: run the glue (add+rms+quantize) and prebuild
          op2's A operands + d DIRECTLY — op2 has NO X fill.

Layout facts the host must honor:
  - C tensor rows: [0..c_rows1) op1's sections (c_rows1 = 8*(b1+2)*16);
    [c_rows1..c_rows1+2048) the residual (host-written each token);
    [9276..9280) the window's header words (u32 K=1 at ELEM-8, u32
    blocks1 at ELEM-4 — host constants, never touched by drains); then
    8 per-column op2 sections of (b2+2)*16 rows.
  - packed2 layout per column: [w element (ELEM B: rms weight bf16,
    K=3 header, blocks1 chunk word) | op2's b2 blocks].
  - The op2 output rows the host reads sit at offset 9280+32 inside
    each column's op2 section (2 glue dummies first, as ever).
"""

ELEM = 18560
K_MAX = 6144
M_INPUT = 16
TILE_K = 2048
FUSED_M1 = 2048  # hidden width — the rms flavor's row count
RMS_ELEM_ROWS = ELEM // 2  # 9280: the rms fill window, in bf16 rows


def chunks_per_tile(k):
    return k // TILE_K


def my_w4gemvu_fused(dev, cols, M1, K1, M2, group_size=32):
    assert cols == 8, "fused design is laid out for 8 columns"
    assert M1 == FUSED_M1, "the rms flavor assumes the 2048 hidden width"
    assert K1 in (2048, 6144) and M2 % 256 == 0, "see design.py ABI"
    assert group_size == 32
    chunks1 = chunks_per_tile(K1)
    blocks1 = (M1 // cols // M_INPUT) * chunks1
    blocks2 = (M2 // cols // M_INPUT) * 1  # op2 is always K=2048

    dtype_in = np.dtype[np.uint8]
    from ml_dtypes import bfloat16
    dtype_out = np.dtype[bfloat16]

    dev_ty = NPU1() if dev == "npu" else NPU2()

    bytes1_per_col = blocks1 * ELEM
    # [w element | op2 blocks] contiguous — one fill, glue-first order.
    bytes2_per_col = (1 + blocks2) * ELEM
    packed1_total = cols * bytes1_per_col
    packed2_total = cols * bytes2_per_col

    c_rows1 = cols * (blocks1 + 2) * M_INPUT
    # sections + residual (M1 bf16) must end below the header words at
    # rows 9276..9279 (host-written constants the window carries).
    assert c_rows1 + M1 <= RMS_ELEM_ROWS - 4, "window overflow"
    section2_rows = (blocks2 + 2) * M_INPUT
    c_total_rows = RMS_ELEM_ROWS + cols * section2_rows

    L1_A_ty = np.ndarray[(ELEM,), dtype_in]
    L1_C_ty = np.ndarray[(M_INPUT,), dtype_out]

    L3_A1_ty = np.ndarray[(packed1_total,), dtype_in]
    L3_A2_ty = np.ndarray[(packed2_total,), dtype_in]
    L3_X_ty = np.ndarray[(ELEM,), dtype_in]
    L3_C_ty = np.ndarray[(c_total_rows,), dtype_out]

    fused_matvec = Kernel(
        "w4gemvu_matvec_bf16",
        "w4gemvu.o",
        [
            np.int32,
            L1_A_ty,
            L1_C_ty,
            np.int32,
            np.int32,
        ],
    )

    A_fifos = [
        ObjectFifo(L1_A_ty, name=f"A_L3L1_{i}", depth=2) for i in range(cols)
    ]
    C_fifos = [
        ObjectFifo(L1_C_ty, name=f"C_L1L3_{i}", depth=2) for i in range(cols)
    ]

    def core_body(A_fifo, C_fifo, fn):
        for _ in range_(0xFFFFFFFF):
            a = A_fifo.acquire(1)
            c = C_fifo.acquire(1)
            fn(M_INPUT, a, c, 32, 0)
            C_fifo.release(1)
            A_fifo.release(1)

    workers = [
        Worker(
            core_body,
            [A_fifos[i].cons(), C_fifos[i].prod(), fused_matvec],
        )
        for i in range(cols)
    ]

    X_taps = [
        TensorAccessPattern(
            tensor_dims=(1, ELEM),
            offset=0,
            sizes=[1, 1, 1, ELEM],
            strides=[0, 0, 0, 1],
        )
        for _ in range(cols)
    ]
    A1_taps = [
        TensorAccessPattern(
            tensor_dims=(1, packed1_total),
            offset=col * bytes1_per_col,
            sizes=[1, 1, 1, bytes1_per_col],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]
    A2_taps = [
        TensorAccessPattern(
            tensor_dims=(1, packed2_total),
            offset=col * bytes2_per_col,
            sizes=[1, 1, 1, bytes2_per_col],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]
    # rms_in: the whole op1-C window as ONE element (bf16 scalars; the
    # 9280-row window == 18560 B == exactly one fifo element).
    RMS_taps = [
        TensorAccessPattern(
            tensor_dims=(1, c_total_rows),
            offset=0,
            sizes=[1, 1, 1, RMS_ELEM_ROWS],
            strides=[0, 0, 0, 1],
        )
        for _ in range(cols)
    ]
    # op1 C drains: per-column sections in the LOW c_rows1 rows.
    C1_taps = [
        TensorAccessPattern(
            tensor_dims=(1, c_total_rows),
            offset=col * (blocks1 + 2) * M_INPUT,
            sizes=[1, 1, 1 + blocks1, M_INPUT],
            strides=[0, 0, M_INPUT, 1],
        )
        for col in range(cols)
    ]
    # op2 C drains: sections after the rms window.
    C2_taps = [
        TensorAccessPattern(
            tensor_dims=(1, c_total_rows),
            offset=RMS_ELEM_ROWS + col * section2_rows,
            sizes=[1, 1, blocks2 + 2, M_INPUT],
            strides=[0, 0, M_INPUT, 1],
        )
        for col in range(cols)
    ]

    rt = Runtime()
    with rt.sequence(L3_A1_ty, L3_A2_ty, L3_X_ty, L3_C_ty) as (
        A1,
        A2,
        X,
        C,
    ):
        rt.start(*workers)
        tg1 = rt.task_group()
        for i in range(cols):
            rt.fill(A_fifos[i].prod(), X, X_taps[i], task_group=tg1)
            rt.fill(A_fifos[i].prod(), A1, A1_taps[i], task_group=tg1)
        for i in range(cols):
            rt.drain(C_fifos[i].cons(), C, C1_taps[i], task_group=tg1, wait=True)
        rt.finish_task_group(tg1)

        tg2 = rt.task_group()
        for i in range(cols):
            rt.fill(A_fifos[i].prod(), C, RMS_taps[i], task_group=tg2)
            rt.fill(A_fifos[i].prod(), A2, A2_taps[i], task_group=tg2)
        for i in range(cols):
            rt.drain(C_fifos[i].cons(), C, C2_taps[i], task_group=tg2, wait=True)
        rt.finish_task_group(tg2)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog="AIE fused rms-pair w4gemvu MLIR design",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M1", type=int, required=True)
    argparser.add_argument("-K1", type=int, required=True)
    argparser.add_argument("-M2", type=int, required=True)
    argparser.add_argument("--cols", type=int, default=8)
    argparser.add_argument("--output-file-path", "-o", type=str, required=True)
    args = argparser.parse_args()
    module = my_w4gemvu_fused(args.dev, args.cols, args.M1, args.K1, args.M2)
    with open(args.output_file_path, "w") as f:
        f.write(str(module))
