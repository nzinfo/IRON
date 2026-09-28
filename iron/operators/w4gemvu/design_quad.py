# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
from pathlib import Path
import argparse

from aie.dialects.aiex import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2

"""M6 QUAD whole-layer exec (P19): pair A+B merged so ONE exec runs

    o -> rms1 -> gateup -> swiglu -> down -> rms2 -> qkv'

which removes the last host round trip inside a layer (the gateup read
back + host swiglu + quantize + residual sync that P18 still paid
between its two pairs). Kernel flavors K=4 (gate window staging) and
K=5 (device swiglu + down A-op prebuild) carry the swiglu; K=1/K=3 run
unchanged for the two rms glues. Shapes are the hy-mt2 layer, fixed:
o 2048x2048, gateup 12288x2048, down 2048x6144, qkv 3072x2048.

C tensor layout (bf16 rows; every window's header words live in pad
rows the drains never touch):
  [0, 9280)     win1 = o sections (8x288) | residual1 @2304 | pad |
                hdr K=1/blocks=16 @9276..9279
  [9280, 15552) gateup sections cols 0..3 (4 x 1568)
  [15552, 18560) padA — K=4 hdr words @18556..18559 (host constants)
  [18560, 24832) gateup sections cols 4..7 — the K=5 window reads
                [18560, 27840): [up cols | padB]
  [24832, 27840) padB — K=5 hdr words @27836..27839
  [27840, 37120) win2 = down sections (8x800) | residual2 @34240 |
                pad | hdr K=2/blocks=48 @37116..37119
  [37120, 40704) qkv sections (8 x 448)

Why padA/padB exist: a C-sourced fill produces WHOLE fifo elements
(9280 rows), and the K=4/K=5 header words sit at the element tail —
they must not land on live gate/up rows, so each half-window is padded
out with host-owned dead rows. The gateup drain therefore writes
column sections with a +3008-row gap for cols >= 4 (per-column tap
offsets — each column has its own tap, the gap is just arithmetic).

Dummy groups ahead of real section rows (each glue element emits a
zero C): gate/up sections 2 dummies (K=1 + K=3), down 2 (K=4 + K=5),
qkv 3 (K=2 + K=1 re-read + K=3). o has 1 (the X element). The qkv
sections are 27 written groups on a 28-group stride. P20: the section
dummies never CROSS a group boundary anymore — tg3/tg4 drain their two
window-fill zero Cs to the padA/padB dead rows within their own group,
C3 covers the 48 down blocks (+2 dummy rows stay at c_init zeros), and
C4 re-consumes its K=3 head's zero C in-group (25 groups at +2 dummies,
block 0 on row 48).

residual2 = x' = x + o_out is NOT host-computable before the exec
(o_out is produced by tg1), so tg4 carries a THIRD fill: a re-read of
the win1 window (o sections + residual1 are still live in C), which
the kernel dispatches via the quad flag K=5 sets — the re-read window
is byte-identical to tg2's, so its header still says K=1. The
resulting h2' lands at the residual slot the K=3 glue already reads.

BD law (P16): every task group keeps <= 4 fills per shim ahead of its
drain. P20 hardened this to the exact proven pair shape: EVERY group
now carries its fills plus its OWN drain (2F+1D max, fills-between-
drains <= 2). The original design let tg3/tg4's K=4/K=5 (K=2/K=1')
zero Cs PARK in the depth-2 C fifo across the task-group boundary
until the next group's drain consumed them as section dummies — and
that cross-group handoff raced on ~1-3% of execs: the run timed out
with a whole-column down or qkv chunk missing, the first missing row
always a section's first data row (P20 fingerprints). The glue zero Cs
now drain to the padA/padB dead rows within their own group.
"""

ELEM = 18560
K_MAX = 6144
M_INPUT = 16
TILE_K = 2048
QUAD_M1 = 2048  # hidden width (both rms flavors)
RMS_ELEM_ROWS = ELEM // 2  # 9280: one C-sourced window, in bf16 rows
PAD_ROWS = 3008  # gate/up half-window pad (hdr-word parking)


def my_w4gemvu_quad(dev, cols, M1, K1, M2, M3, K3, M4, group_size=32):
    assert cols == 8, "quad design is laid out for 8 columns"
    assert (M1, K1, M2, M3, K3, M4) == (
        QUAD_M1,
        2048,
        12288,
        QUAD_M1,
        6144,
        3072,
    ), "quad is the fixed hy-mt2 layer shape"
    assert group_size == 32

    blocks_o = 16  # o:        (2048/8/16) x 1 chunk
    blocks_gu = 96  # gateup:   (12288/8/16) x 1
    blocks_dn = 48  # down:     (2048/8/16) x 3 chunks
    blocks_q = 24  # qkv:      (3072/8/16) x 1

    dtype_in = np.dtype[np.uint8]
    from ml_dtypes import bfloat16
    dtype_out = np.dtype[bfloat16]

    dev_ty = NPU1() if dev == "npu" else NPU2()

    # P21-2: packed1 is FRONT-GROUPED — [X0..X7 | blocks_col0..col7] —
    # not the P12 per-column [X | blocks] stream. The activation still
    # rides packed1 (no 6th BO: the ctrl kernel signature is capped at 5
    # buffer args by aiecc's emit_design_kernel_json), but the 8 X
    # elements sit CONTIGUOUSLY at the BO head so the host's per-exec
    # dirty set is one 148KB run: 1 SYNC_BO + ~10us instead of flushing
    # the whole 2.5MB weight BO behind 8 strided heads (P21-1 measured
    # both whole-BO and 8-region syncs at 129-154us/layer). Each column
    # therefore takes TWO fills in tg1 (its X element, then its blocks —
    # the same 2-fill-per-column shape tg2 already proves). Kernel-side
    # nothing changes: elements self-describe via tail headers, so the
    # per-column fifo stream is still [X | 16 blocks].
    bytes1_per_col = (1 + blocks_o) * ELEM  # total per column, layout above
    x_region = cols * ELEM  # front: X0..X7 contiguous
    blocks1_off = lambda col: x_region + col * blocks_o * ELEM
    bytes2_per_col = (1 + blocks_gu) * ELEM  # K=3 w element first
    bytes3_per_col = blocks_dn * ELEM  # no X, no head — pure blocks
    bytes4_per_col = (1 + blocks_q) * ELEM  # K=3 w element first
    packed1_total = cols * bytes1_per_col
    packed2_total = cols * bytes2_per_col
    packed3_total = cols * bytes3_per_col
    packed4_total = cols * bytes4_per_col

    sec_o = (blocks_o + 2) * M_INPUT  # 288
    sec_gu = (blocks_gu + 2) * M_INPUT  # 1568
    sec_dn = (blocks_dn + 2) * M_INPUT  # 800
    sec_q = (blocks_q + 3 + 1) * M_INPUT  # 448: 27 written + 1 pad
    GATE_OFF = RMS_ELEM_ROWS  # 9280
    UP_OFF = GATE_OFF + 4 * sec_gu + PAD_ROWS  # 18560
    WIN2_OFF = UP_OFF + 4 * sec_gu + PAD_ROWS  # 27840
    QKV_OFF = WIN2_OFF + RMS_ELEM_ROWS  # 37120
    c_total_rows = QKV_OFF + cols * sec_q  # 40704

    L1_A_ty = np.ndarray[(ELEM,), dtype_in]
    L1_C_ty = np.ndarray[(M_INPUT,), dtype_out]

    L3_A1_ty = np.ndarray[(packed1_total,), dtype_in]
    L3_A2_ty = np.ndarray[(packed2_total,), dtype_in]
    L3_A3_ty = np.ndarray[(packed3_total,), dtype_in]
    L3_A4_ty = np.ndarray[(packed4_total,), dtype_in]
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

    # P21-2: per column the X element (front region) and the o blocks
    # (behind the whole X region) are two fills — stream order per fifo
    # stays [X | 16 blocks].
    A1x_taps = [
        TensorAccessPattern(
            tensor_dims=(1, packed1_total),
            offset=col * ELEM,
            sizes=[1, 1, 1, ELEM],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]
    A1b_taps = [
        TensorAccessPattern(
            tensor_dims=(1, packed1_total),
            offset=blocks1_off(col),
            sizes=[1, 1, 1, blocks_o * ELEM],
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
    A3_taps = [
        TensorAccessPattern(
            tensor_dims=(1, packed3_total),
            offset=col * bytes3_per_col,
            sizes=[1, 1, 1, bytes3_per_col],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]
    A4_taps = [
        TensorAccessPattern(
            tensor_dims=(1, packed4_total),
            offset=col * bytes4_per_col,
            sizes=[1, 1, 1, bytes4_per_col],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]
    # C-sourced windows: one whole fifo element (9280 bf16 rows).
    win1_taps = [
        TensorAccessPattern(
            tensor_dims=(1, c_total_rows),
            offset=0,
            sizes=[1, 1, 1, RMS_ELEM_ROWS],
            strides=[0, 0, 0, 1],
        )
        for _ in range(cols)
    ]
    swa_taps = [
        TensorAccessPattern(
            tensor_dims=(1, c_total_rows),
            offset=GATE_OFF,
            sizes=[1, 1, 1, RMS_ELEM_ROWS],
            strides=[0, 0, 0, 1],
        )
        for _ in range(cols)
    ]
    swb_taps = [
        TensorAccessPattern(
            tensor_dims=(1, c_total_rows),
            offset=UP_OFF,
            sizes=[1, 1, 1, RMS_ELEM_ROWS],
            strides=[0, 0, 0, 1],
        )
        for _ in range(cols)
    ]
    win2_taps = [
        TensorAccessPattern(
            tensor_dims=(1, c_total_rows),
            offset=WIN2_OFF,
            sizes=[1, 1, 1, RMS_ELEM_ROWS],
            strides=[0, 0, 0, 1],
        )
        for _ in range(cols)
    ]
    # o C drains: one X dummy group ahead of the 16 real groups.
    C1_taps = [
        TensorAccessPattern(
            tensor_dims=(1, c_total_rows),
            offset=col * sec_o,
            sizes=[1, 1, 1 + blocks_o, M_INPUT],
            strides=[0, 0, M_INPUT, 1],
        )
        for col in range(cols)
    ]
    # gateup C drains: 2 dummy groups; cols >= 4 jump the padA gap.
    C2_taps = [
        TensorAccessPattern(
            tensor_dims=(1, c_total_rows),
            offset=(
                GATE_OFF + col * sec_gu
                if col < 4
                else UP_OFF + (col - 4) * sec_gu
            ),
            sizes=[1, 1, blocks_gu + 2, M_INPUT],
            strides=[0, 0, M_INPUT, 1],
        )
        for col in range(cols)
    ]
    # down C drains: real blocks only, +2 dummy rows of section skip —
    # P20: the K=4/K=5 zero Cs are drained by tg3 itself (below), so the
    # dummy rows are never written (c_init zeros, matching the zeros the
    # parked dummy groups used to leave there).
    C3_taps = [
        TensorAccessPattern(
            tensor_dims=(1, c_total_rows),
            offset=WIN2_OFF + col * sec_dn + 2 * M_INPUT,
            sizes=[1, 1, blocks_dn, M_INPUT],
            strides=[0, 0, M_INPUT, 1],
        )
        for col in range(cols)
    ]
    # qkv C drains: the K=2/K=1' zero Cs go with tg4's own glue drain;
    # the K=3 #2 head element rides the A4 fill (filled AND consumed
    # within tg4b — the same in-group head pattern C2 uses), so this
    # drain pops [K=3 zero | 24 blocks] = 25 groups at +2 dummy groups,
    # landing block 0 on row 48 where the host readback expects it.
    C4_taps = [
        TensorAccessPattern(
            tensor_dims=(1, c_total_rows),
            offset=QKV_OFF + col * sec_q + 2 * M_INPUT,
            sizes=[1, 1, blocks_q + 1, M_INPUT],
            strides=[0, 0, M_INPUT, 1],
        )
        for col in range(cols)
    ]
    # P20: tg3/tg4's own drains for the glue zero Cs (2 groups x 16 rows
    # per column), landed in the padA/padB interiors — element payload the
    # kernels never read (headers sit at the very pad tail) and no other
    # tap touches. Restores the proven pair choreography: no C element
    # ever crosses a task-group boundary inside the fifo.
    GLUE_C3_OFF = GATE_OFF + 4 * sec_gu + 48  # 15600, padA interior
    GLUE_C4_OFF = UP_OFF + 4 * sec_gu + 48  # 24880, padB interior
    glue3_taps = [
        TensorAccessPattern(
            tensor_dims=(1, c_total_rows),
            offset=GLUE_C3_OFF + col * 2 * M_INPUT,
            sizes=[1, 1, 2, M_INPUT],
            strides=[0, 0, M_INPUT, 1],
        )
        for col in range(cols)
    ]
    glue4_taps = [
        TensorAccessPattern(
            tensor_dims=(1, c_total_rows),
            offset=GLUE_C4_OFF + col * 2 * M_INPUT,
            sizes=[1, 1, 2, M_INPUT],
            strides=[0, 0, M_INPUT, 1],
        )
        for col in range(cols)
    ]

    rt = Runtime()
    with rt.sequence(
        L3_A1_ty, L3_A2_ty, L3_A3_ty, L3_A4_ty, L3_C_ty
    ) as (
        A1,
        A2,
        A3,
        A4,
        C,
    ):
        rt.start(*workers)
        # tg1: o (P21-2: per column [X fill | blocks fill] from the
        # front-grouped packed1 — tg2's 2-fill-per-column loop shape) -> C1
        tg1 = rt.task_group()
        for i in range(cols):
            rt.fill(A_fifos[i].prod(), A1, A1x_taps[i], task_group=tg1)
            rt.fill(A_fifos[i].prod(), A1, A1b_taps[i], task_group=tg1)
        for i in range(cols):
            rt.drain(C_fifos[i].cons(), C, C1_taps[i], task_group=tg1, wait=True)
        rt.finish_task_group(tg1)

        # tg2: rms1 window (from C) + [K=3 w element | 96 gateup blocks]
        tg2 = rt.task_group()
        for i in range(cols):
            rt.fill(A_fifos[i].prod(), C, win1_taps[i], task_group=tg2)
            rt.fill(A_fifos[i].prod(), A2, A2_taps[i], task_group=tg2)
        for i in range(cols):
            rt.drain(C_fifos[i].cons(), C, C2_taps[i], task_group=tg2, wait=True)
        rt.finish_task_group(tg2)

        # tg3: gate window (K=4) + up window (K=5), each glue zero C
        # drained HERE to the padA dead rows (P20 — see glue3_taps; they
        # used to park in the C fifo for tg3b's drain, the ~1-3%/exec
        # cross-group hang). down ships NO X and NO head: K=5 prebuilt
        # its A operands.
        tg3 = rt.task_group()
        for i in range(cols):
            rt.fill(A_fifos[i].prod(), C, swa_taps[i], task_group=tg3)
            rt.fill(A_fifos[i].prod(), C, swb_taps[i], task_group=tg3)
        for i in range(cols):
            rt.drain(C_fifos[i].cons(), C, glue3_taps[i], task_group=tg3, wait=True)
        rt.finish_task_group(tg3)

        tg3b = rt.task_group()
        for i in range(cols):
            rt.fill(A_fifos[i].prod(), A3, A3_taps[i], task_group=tg3b)
        for i in range(cols):
            rt.drain(C_fifos[i].cons(), C, C3_taps[i], task_group=tg3b, wait=True)
        rt.finish_task_group(tg3b)

        # tg4: win2 (K=2, down partials + residual2) + the win1 re-read
        # (K=1 + quad flag -> stage1r stages h2' = x + o_out over the
        # residual slot), the K=2/K=1' zero Cs drained here to padB
        # (glue4_taps, same P20 restructure).
        tg4 = rt.task_group()
        for i in range(cols):
            rt.fill(A_fifos[i].prod(), C, win2_taps[i], task_group=tg4)
            rt.fill(A_fifos[i].prod(), C, win1_taps[i], task_group=tg4)
        for i in range(cols):
            rt.drain(C_fifos[i].cons(), C, glue4_taps[i], task_group=tg4, wait=True)
        rt.finish_task_group(tg4)

        tg4b = rt.task_group()
        for i in range(cols):
            rt.fill(A_fifos[i].prod(), A4, A4_taps[i], task_group=tg4b)
        for i in range(cols):
            rt.drain(C_fifos[i].cons(), C, C4_taps[i], task_group=tg4b, wait=True)
        rt.finish_task_group(tg4b)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog="AIE quad whole-layer w4gemvu MLIR design",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M1", type=int, required=True)
    argparser.add_argument("-K1", type=int, required=True)
    argparser.add_argument("-M2", type=int, required=True)
    argparser.add_argument("-M3", type=int, required=True)
    argparser.add_argument("-K3", type=int, required=True)
    argparser.add_argument("-M4", type=int, required=True)
    argparser.add_argument("--cols", type=int, default=8)
    argparser.add_argument("--output-file-path", "-o", type=str, required=True)
    args = argparser.parse_args()
    module = my_w4gemvu_quad(
        args.dev, args.cols, args.M1, args.K1, args.M2, args.M3, args.K3, args.M4
    )
    with open(args.output_file_path, "w") as f:
        f.write(str(module))
