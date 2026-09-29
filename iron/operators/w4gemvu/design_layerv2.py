# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.

"""P28 layer-v2: the WHOLE transformer layer in ONE task group per exec.

P27-4 established that the task group is the atom of scheduling cost
(129 groups x ~75us = the entire device-side gap to FLM) and that groups
can only be REMOVED by designing away cross-group dependencies. This
design is that removal (P28-1 persistent-worker model, P28-3 ring proof):

    N workers (N = 8/16, P28-6 ring widening; 32 is past the npu2 shim
    channel budget -- see my_layerv2) on tiles (c, 2..5) for c in
    0..N/4 (worker w lives at tile col c = w>>2, row 2+(w&3)), linked
    in a ring whose EVERY edge joins unit-step-neighbor tiles --
    core<->core objectFifos only lower to shared memory between
    mem-affine (adjacent) tiles, anything farther splits into a core
    mem-DMA channel and npu2 cores have just 2 output channels
    (CORE<->CORE OBJECTFIFO LAW, P28-6; see ring_tables/HAM16):
    N=8:  order 0,1,2,3,7,6,5,4 (the two-column serpentine)
    N=16: order 0,1,2,3,7,6,5,9,10,11,15,14,13,12,8,4 (Hamiltonian)

The per-worker geometry is derived from N (rows = 2048/N residual rows,
jpw = 6144/N gate/up rows -- all exact for power-of-2 N <= 128):

    N=8:  rows 256, N_O 16,  gate/up 48, down 48, qkv 24, xn1 16, W 186
    N=16: rows 128, N_O 8,   gate/up 24, down 24, qkv 12, xn1 8,  W 94

Per exec each worker consumes N_WELEM+2 A elements in ONE stream (fifo
order is the only barrier needed -- no windows, no C round trips):

    [X(K=0) | xn(K=100) | o xN_O | w2(K=101) | gate xN_GATE | up xN_UP |
     down xN_DOWN | w1(K=102) | qkv xN_QKV]

and produces N_QKV+N_CXN C elements (qkv + xn1 drain). The kernel reads
N from the X element ([6404,6408) u32) -- ONE .o serves every width
(w4gemvu_layer.cc P28-6). The three all-gathers (o-out x', swiglu sw,
down-out xn1) ride core<->core ObjectFifos around the ring instead of
C->DDR->window fills:

    ring1 (r13 fifo, rows*2B): x' gather -- fr1 makes bf16(xn+o) and
        also stores the OWN slot; rounds 1..N-2 store+forward, round
        N-1 stores.
    ring2 (r2 fifo, jpw + align32(grp*2) bytes): sw gather -- fr2 packs
        own jpw int8 q + grp u16 scales; same round structure.
    ring3 (REUSES the r13 fifo): xn1 gather -- fr3 folds dacc into the
        own x' slot in place; same round structure; lv_fw1 forwards.

Everything else (arena, rms, swiglu, down accumulation) lives in the
kernel's .bss -- see aie_kernels/aie2p/w4gemvu_layer.cc for the K
flavors and the numerics chain (bit-mirrors the v5/fused golden).

PMEM law (P28-4, still binding): aiecc's opt unrolls small-trip-count
scf.for loops (empirically <=16 trips) and the wrapper must stay rolled.
Every section below runs BARE when its element count is >= 48 (proven
rolled) and as a guarded range_(48) loop otherwise; ring rounds are
N-1 <= 31 < 48 so the guarded form always fits.

The 4-arg w4gemvu_layer_a entry serves every flavor that acquires no C
element (MLIR cannot pass a null memref); only the qkv phase of K=2048
uses the 5-arg w4gemvu_layer_bf16. Eleven Kernel objects share the one
bin_name -- the Worker dedups bin_names, so the .o links once (the mha
design's proven pattern).

rt.sequence(W, X, XN, C) = 4 tensor BOs, inside the 5-BO ctrl regmap
limit (P19b law). X carries the per-exec attn operands + worker id at
byte 6400 + worker count N at 6404; XN carries the residual (the
previous exec's xn1 drain BO in the E2E integration); W is one linear
N_WELEM-element run per worker.
"""

import numpy as np
from ml_dtypes import bfloat16
import argparse

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_, if_
from aie.extras.dialects.arith import cmpi, constant, subi
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2

ELEM = 18560          # A fifo element (v5 ABI: 16-row x 2048-k tile)
M_INPUT = 16          # C fifo element = 16 bf16 (32B)
HIDDEN = 2048
INTER = 6144
QKV_M = 3072


# N=16 ring order: the all-adjacent Hamiltonian cycle over the 4x4
# worker grid. CORE<->CORE OBJECTFIFO LAW (P28-6, source-proven in
# mlir-aie AIEObjectFifoStatefulTransform): a core-to-core objectFifo
# lowers to zero-flow shared memory + locks IFF the endpoints satisfy
# isLegalMemAffinity (unit-step neighbors, horizontal or vertical --
# both directions proven by the N=8 board binary); a non-adjacent edge
# SPLITS and takes one mem-DMA MM2S channel on the producer core tile,
# and npu2 cores have only 2 output channels. The N=16 SERPENTINE wrap
# edge 12 -> 0 = tile (3,2) -> (0,2) spans 3 columns, and worker 12
# produces BOTH wrap fifos (ring13_12, ring2_12) plus its C stream =
# 3 MM2S > 2 -> "'aie.tile' op number of output DMA channel exceeded!".
# A Hamiltonian cycle with every edge (incl. the wrap) a unit step
# exists on any grid with an even side; this one keeps the N=8 prefix
# 0,1,2,3,7,6,5 and wraps 4 -> 0 (tiles (1,2) -> (0,2), adjacent).
HAM16 = [0, 1, 2, 3, 7, 6, 5, 9, 10, 11, 15, 14, 13, 12, 8, 4]


def ring_tables(n):
    """Ring order/SUCC over the n-worker grid (n/4 columns x 4 rows,
    workers listed column-major: w = c*4 + r). N=8: the classic
    serpentine (every edge adjacent by construction -- cols 0-1 only).
    N=16: HAM16 (see above). Mirrored by the kernel's lv_xelem (the
    position arithmetic there) and by the host packer/golden --
    three-way contract."""
    assert n % 4 == 0 and n >= 8
    if n == 16:
        order = list(HAM16)
    else:
        order = []
        for c in range(n // 4):
            rr = range(4) if c % 2 == 0 else reversed(range(4))
            order += [c * 4 + r for r in rr]
    succ = {order[i]: order[(i + 1) % n] for i in range(n)}
    return order, succ


def my_layerv2(dev, cols=8):
    """cols = WORKER COUNT (8/16), not shim columns. 32 is impossible on
    npu2: 8 shims x 2 MM2S = 16 inbound channels and one A fifo per
    worker already spends them all at N=16 (SHIM CHANNEL LAW, P28-6)."""
    n = cols
    assert n in (8, 16), "npu2 shim budget: 8 shims x 2 ch = 16 streams = N=16 max"
    rows = HIDDEN // n            # residual rows per worker
    N_O = rows // 16
    N_GATE = (INTER // n) // 16
    N_UP = N_GATE
    N_DOWN = 3 * N_O              # rows x 3 K-chunks
    N_QKV = (QKV_M // n) // 16
    N_CXN = N_O
    N_WELEM = N_O + N_GATE + N_UP + N_DOWN + N_QKV + 2  # + w2 + w1
    grp = (INTER // n) // 32      # quant groups per worker
    R13 = rows                    # bf16 per ring1/ring3 element
    R2 = (INTER // n) + ((grp * 2 + 31) // 32) * 32  # q + align32(scales)
    ROUNDS = n - 1                # ring gather rounds
    _, SUCC = ring_tables(n)
    PRED = {v: k for k, v in SUCC.items()}

    dtype_in = np.dtype[np.uint8]
    dtype_out = np.dtype[bfloat16]

    dev_ty = NPU1() if dev == "npu" else NPU2()

    L1_A_ty = np.ndarray[(ELEM,), dtype_in]
    L1_C_ty = np.ndarray[(M_INPUT,), dtype_out]
    L1_R13_ty = np.ndarray[(R13,), dtype_out]
    L1_R2_ty = np.ndarray[(R2,), dtype_in]

    L3_W_ty = np.ndarray[(n * N_WELEM * ELEM,), dtype_in]
    L3_X_ty = np.ndarray[(n * ELEM,), dtype_in]
    L3_XN_ty = np.ndarray[(n * ELEM,), dtype_in]
    L3_C_ty = np.ndarray[(n * (N_QKV + N_CXN) * M_INPUT,), dtype_out]

    bin_name = "w4gemvu_layer.o"
    # Signature-trimmed entries (PMEM law): m/group_size/tile_idx were
    # ignored by the kernel -- every trimmed arg is a constant the wrapper
    # materializes at every call site.
    k_bf = Kernel(
        "w4gemvu_layer_bf16",
        bin_name,
        [L1_A_ty, L1_C_ty],
    )
    k_a = Kernel(
        "w4gemvu_layer_a",
        bin_name,
        [L1_A_ty],
    )
    k_fr1 = Kernel("lv_fr1", bin_name, [L1_R13_ty])
    k_fw1 = Kernel("lv_fw1", bin_name, [L1_R13_ty, L1_R13_ty])
    k_st1 = Kernel("lv_st1", bin_name, [L1_R13_ty])
    k_fr2 = Kernel("lv_fr2", bin_name, [L1_R2_ty])
    k_fw2 = Kernel("lv_fw2", bin_name, [L1_R2_ty, L1_R2_ty])
    k_st2 = Kernel("lv_st2", bin_name, [L1_R2_ty])
    k_fr3 = Kernel("lv_fr3", bin_name, [L1_R13_ty])
    k_st3 = Kernel("lv_st3", bin_name, [L1_R13_ty])
    k_cxn = Kernel("lv_cxn", bin_name, [L1_C_ty])

    A_fifos = [
        ObjectFifo(L1_A_ty, name=f"A_L3L1_{i}", depth=2) for i in range(n)
    ]
    C_fifos = [
        ObjectFifo(L1_C_ty, name=f"C_L1L3_{i}", depth=2) for i in range(n)
    ]
    # ring13_fifos[w] carries the edge w -> SUCC[w] (ring1 AND ring3, which
    # reuse the same fifo sequentially -- each ring phase pushes and
    # consumes exactly ROUNDS elements per edge, so the edge is empty
    # between phases and the next phase self-starts).
    ring13_fifos = [
        ObjectFifo(L1_R13_ty, name=f"ring13_{w}", depth=2) for w in range(n)
    ]
    ring2_fifos = [
        ObjectFifo(L1_R2_ty, name=f"ring2_{w}", depth=2) for w in range(n)
    ]

    def core_body(a_f, r13_out, r13_in, r2_out, r2_in, c_f,
                  kbf, ka, kfr1, kfw1, kst1, kfr2, kfw2, kst2, kfr3, kst3, kcxn):
        # PMEM law (P28-4): every loop is either BARE with >= 48 trips
        # (proven rolled) or a guarded range_(48) loop (rolled with an
        # scf.if branch selecting the live iterations -- the wasted guard
        # iterations are pure local compare/branch, no fifo traffic).
        def lt(i, n_):
            return cmpi("slt", i, constant(n_, index=True))

        # consume `count` A elements through the no-C entry, guarded to
        # stay rolled when count < 48.
        def consume(count):
            if count >= 48:
                for _ in range_(count):
                    a = a_f.acquire(1)
                    ka(a)
                    a_f.release(1)
            else:
                for i in range_(48):
                    with if_(lt(i, count), hasElse=False):
                        a = a_f.acquire(1)
                        ka(a)
                        a_f.release(1)

        # one ring gather (make + ROUNDS receive rounds; the last round
        # stores only -- its chunk would return to its origin).
        def gather(r_out, r_in, kfr, kfw, kst):
            ob = r_out.acquire(1)
            kfr(ob)
            r_out.release(1)
            for i in range_(48):
                with if_(lt(i, ROUNDS), hasElse=False):
                    s = r_in.acquire(1)
                    kst(s)
                    with if_(lt(i, ROUNDS - 1), hasElse=False):
                        ob = r_out.acquire(1)
                        kfw(s, ob)
                        r_out.release(1)
                    r_in.release(1)

        for _ in range_(0xFFFFFFFF):
            # [X(K=0) | xn(K=100) | o xN_O] (phase O -> lv_o slices; the
            # X element resets all per-exec state incl. the N geometry).
            consume(2 + N_O)
            # ---- ring 1: gather x' = bf16(xn + o) over all N slots ----
            gather(r13_out, r13_in, kfr1, kfw1, kst1)
            # [w2(K=101) | gate xN_GATE | up xN_UP] (rms resets gate/up
            # counters; every 2nd up block runs one swiglu group).
            consume(1 + N_GATE + N_UP)
            # ---- ring 2: gather sw (int8 q + scales) ----
            gather(r2_out, r2_in, kfr2, kfw2, kst2)
            # [down xN_DOWN] (c-major chunk-blocks accumulate f32 into
            # dacc).
            consume(N_DOWN)
            # ---- ring 3: gather xn1 = bf16(x' + dacc) in place ----
            # MUST complete before w1: K=102's rms reads the gathered
            # xn1 out of lv_shared.
            gather(r13_out, r13_in, kfr3, kfw1, kst3)
            # [w1(K=102) | qkv xN_QKV | cxn xN_CXN], one guarded loop,
            # stream order preserved: w1's rms consumes the gathered
            # xn1, sets phase = QKV and rebuilds the arena; qkv blocks
            # then write the REAL C elements; cxn drains the own xn1
            # chunk as C (next exec's residual source). Sections are
            # sibling guards -- an unsigned range check
            # 0 <= i-base < n (a signed lt fires on the negative wrap)
            # instead of nested if/else.
            def rng(i, base, n_):
                return cmpi(
                    "ult",
                    subi(i, constant(base, index=True)),
                    constant(n_, index=True),
                )

            for i in range_(48):
                with if_(lt(i, 1 + N_QKV + N_CXN), hasElse=False):
                    with if_(lt(i, 1), hasElse=False):
                        a = a_f.acquire(1)
                        ka(a)  # w1: no C element
                        a_f.release(1)
                    with if_(rng(i, 1, N_QKV), hasElse=False):
                        a = a_f.acquire(1)
                        c = c_f.acquire(1)
                        kbf(a, c)
                        c_f.release(1)
                        a_f.release(1)
                    with if_(rng(i, 1 + N_QKV, N_CXN), hasElse=False):
                        c = c_f.acquire(1)
                        kcxn(c)
                        c_f.release(1)

    workers = [
        Worker(
            core_body,
            [
                A_fifos[w].cons(),
                ring13_fifos[w].prod(),
                ring13_fifos[PRED[w]].cons(),
                ring2_fifos[w].prod(),
                ring2_fifos[PRED[w]].cons(),
                C_fifos[w].prod(),
                k_bf,
                k_a,
                k_fr1,
                k_fw1,
                k_st1,
                k_fr2,
                k_fw2,
                k_st2,
                k_fr3,
                k_st3,
                k_cxn,
            ],
        )
        for w in range(n)
    ]

    X_taps = [
        TensorAccessPattern(
            tensor_dims=(1, n * ELEM),
            offset=w * ELEM,
            sizes=[1, 1, 1, ELEM],
            strides=[0, 0, 0, 1],
        )
        for w in range(n)
    ]
    XN_taps = X_taps  # identical geometry, different tensor
    W_taps = [
        TensorAccessPattern(
            tensor_dims=(1, n * N_WELEM * ELEM),
            offset=w * N_WELEM * ELEM,
            sizes=[1, 1, 1, N_WELEM * ELEM],
            strides=[0, 0, 0, 1],
        )
        for w in range(n)
    ]
    # C drain per worker: N_QKV+N_CXN elements linear (qkv then xn1).
    # The host splits the two sections by ring position.
    C_taps = [
        TensorAccessPattern(
            tensor_dims=(1, n * (N_QKV + N_CXN) * M_INPUT),
            offset=w * (N_QKV + N_CXN) * M_INPUT,
            sizes=[1, 1, N_QKV + N_CXN, M_INPUT],
            strides=[0, 0, M_INPUT, 1],
        )
        for w in range(n)
    ]

    rt = Runtime()
    with rt.sequence(L3_W_ty, L3_X_ty, L3_XN_ty, L3_C_ty) as (W, X, XN, C):
        rt.start(*workers)
        tg = rt.task_group()
        # P28-6 FILL-ORDER LAW: every worker's small unblocking fills
        # (X, XN) before ANY huge W fill. At N=16 two workers share a
        # shim; interleaved, a worker's XN can queue behind its
        # roommate's 1.74 MB W fill while the roommate parks at the
        # first ring gather with that fill half-streamed -- cross-worker
        # shim starvation -> ring deadlock. N=8 (1 worker/shim) is
        # immune either way; see notes/perf-lab.md 6f-1.
        for w in range(n):
            rt.fill(A_fifos[w].prod(), X, X_taps[w], task_group=tg)
            rt.fill(A_fifos[w].prod(), XN, XN_taps[w], task_group=tg)
        for w in range(n):
            rt.fill(A_fifos[w].prod(), W, W_taps[w], task_group=tg)
        for w in range(n):
            rt.drain(
                C_fifos[w].cons(),
                C,
                C_taps[w],
                task_group=tg,
                wait=True,
            )
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(prog="P28 layer-v2")
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("--cols", type=int, default=8,
                           help="worker count: 8/16/32")
    argparser.add_argument("--output-file-path", "-o", type=str, required=True)
    args = argparser.parse_args()
    module = my_layerv2(args.dev, args.cols)
    with open(args.output_file_path, "w") as f:
        f.write(str(module))
