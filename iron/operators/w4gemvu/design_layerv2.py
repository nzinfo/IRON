# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.

"""P28 layer-v2: the WHOLE transformer layer in ONE task group per exec.

P27-4 established that the task group is the atom of scheduling cost
(129 groups x ~75us = the entire device-side gap to FLM) and that groups
can only be REMOVED by designing away cross-group dependencies. This
design is that removal (P28-1 persistent-worker model, P28-3 ring proof):

    8 workers on tiles (0,2)..(0,5),(1,2)..(1,5), serpentine ring
    SUCC = {0:1, 1:2, 2:3, 3:7, 7:6, 6:5, 5:4, 4:0}
    ring position p = (w < 4) ? w : 11 - w

Per exec each worker consumes 188 A elements in ONE stream (fifo order
is the only barrier needed -- no windows, no C round trips):

    [X(K=0) | xn(K=100) | o x16 | w2(K=101) | gate x48 | up x48 |
     down x48 | w1(K=102) | qkv x24]

and produces 40 C elements (24 qkv + 16 xn1 drain). The three
all-gathers (o-out x', swiglu sw, down-out xn1) ride core<->core
ObjectFifos around the ring instead of C->DDR->window fills:

    ring1 (r13 fifo, 512B): x' gather -- fr1 makes bf16(xn+o) and also
        stores the OWN slot; rounds 1..6 store+forward, round 7 stores.
    ring2 (r2 fifo, 832B): sw gather -- fr2 packs own 768 int8 q +
        48B scales; same round structure.
    ring3 (REUSES the r13 fifo): xn1 gather -- fr3 folds dacc into the
        own x' slot in place; same round structure; lv_fw1 forwards.

Everything else (arena, rms, swiglu, down accumulation) lives in the
kernel's .bss -- see aie_kernels/aie2p/w4gemvu_layer.cc for the K
flavors and the numerics chain (bit-mirrors the v5/fused golden).

The 4-arg w4gemvu_layer_a entry serves every flavor that acquires no C
element (MLIR cannot pass a null memref); only the qkv phase of K=2048
uses the 5-arg w4gemvu_layer_bf16. Eleven Kernel objects share the one
bin_name -- the Worker dedups bin_names, so the .o links once (the mha
design's proven pattern).

rt.sequence(W, X, XN, C) = 4 tensor BOs, inside the 5-BO ctrl regmap
limit (P19b law). X carries the per-exec attn operands + worker id at
byte 6400; XN carries the residual (the previous exec's xn1 drain BO in
the E2E integration); W is one linear 186-element run per worker.
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
GS = 32               # int4/int8 quant group size
N_O = 16              # o blocks per worker (256 rows / 16)
N_GATE = 48           # gate blocks per worker (768 rows / 16)
N_UP = 48
N_DOWN = 48           # down blocks per worker (256 rows x 3 chunks / 16)
N_QKV = 24            # qkv blocks per worker (384 rows / 16)
N_CXN = 16            # xn1 drain elements per worker
N_WBLK = N_O + N_GATE + N_UP + N_DOWN + N_QKV  # 184 weight blocks
N_WELEM = N_WBLK + 2  # + w2 + w1 norm elements = 186
N_AELEM = N_WELEM + 2  # + X + xn = 188
R13 = 256             # bf16 per ring1/ring3 element (512B)
R2 = 832              # bytes per ring2 element (768 int8 + 48B scales + pad)
SUCC = {0: 1, 1: 2, 2: 3, 3: 7, 7: 6, 6: 5, 5: 4, 4: 0}
PRED = {v: k for k, v in SUCC.items()}


def my_layerv2(dev, cols=8):
    assert cols == 8, "serpentine tables are for the 8-core npu2 partition"

    dtype_in = np.dtype[np.uint8]
    dtype_out = np.dtype[bfloat16]

    dev_ty = NPU1() if dev == "npu" else NPU2()

    L1_A_ty = np.ndarray[(ELEM,), dtype_in]
    L1_C_ty = np.ndarray[(M_INPUT,), dtype_out]
    L1_R13_ty = np.ndarray[(R13,), dtype_out]
    L1_R2_ty = np.ndarray[(R2,), dtype_in]

    L3_W_ty = np.ndarray[(cols * N_WELEM * ELEM,), dtype_in]
    L3_X_ty = np.ndarray[(cols * ELEM,), dtype_in]
    L3_XN_ty = np.ndarray[(cols * ELEM,), dtype_in]
    L3_C_ty = np.ndarray[(cols * (N_QKV + N_CXN) * M_INPUT,), dtype_out]

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
        ObjectFifo(L1_A_ty, name=f"A_L3L1_{i}", depth=2) for i in range(cols)
    ]
    C_fifos = [
        ObjectFifo(L1_C_ty, name=f"C_L1L3_{i}", depth=2) for i in range(cols)
    ]
    # ring13_fifos[w] carries the edge w -> SUCC[w] (ring1 AND ring3, which
    # reuse the same fifo sequentially -- each ring phase pushes and
    # consumes exactly 7 elements per edge, so the edge is empty between
    # phases and the next phase self-starts).
    ring13_fifos = [
        ObjectFifo(L1_R13_ty, name=f"ring13_{w}", depth=2) for w in range(cols)
    ]
    ring2_fifos = [
        ObjectFifo(L1_R2_ty, name=f"ring2_{w}", depth=2) for w in range(cols)
    ]

    def core_body(a_f, r13_out, r13_in, r2_out, r2_in, c_f,
                  kbf, ka, kfr1, kfw1, kst1, kfr2, kfw2, kst2, kfr3, kst3, kcxn):
        # PMEM law (P28-4): aiecc's LLVM opt step (default<O2>) FULLY
        # UNROLLS small static-trip-count scf.for loops in this wrapper --
        # empirically <=16-trip loops unroll, 48-trip loops stay rolled --
        # and 22 textual kernel call sites x unrolling blew the 16KB
        # program memory (wrapper alone 10896B). Every loop below is
        # therefore 48+ trips with an scf.if guard selecting the live
        # iterations: a rolled loop with a branch, never an unrolled body.
        # The wasted guard iterations are pure local compare/branch -- no
        # fifo traffic. Sections with >=48 real elements run bare.
        def lt(i, n):
            return cmpi("slt", i, constant(n, index=True))

        for _ in range_(0xFFFFFFFF):
            # [X(K=0) | xn(K=100) | o x16] = 18 live of 48 (phase O ->
            # lv_o slices; the X element resets all per-exec state).
            for i in range_(48):
                with if_(lt(i, 18), hasElse=False):
                    a = a_f.acquire(1)
                    ka(a)
                    a_f.release(1)
            # ---- ring 1: gather x' = bf16(xn + o) over all 8 slots ----
            # make first (producer acquire only waits for a free buffer,
            # and the edge is empty at the exec boundary -> self-starts);
            # rounds 1..6 store+forward, round 7 store only (its chunk
            # would return to its origin).
            ob = r13_out.acquire(1)
            kfr1(ob)
            r13_out.release(1)
            for i in range_(48):
                with if_(lt(i, 7), hasElse=False):
                    s = r13_in.acquire(1)
                    kst1(s)
                    with if_(lt(i, 6), hasElse=False):
                        ob = r13_out.acquire(1)
                        kfw1(s, ob)
                        r13_out.release(1)
                    r13_in.release(1)
            # [w2(K=101) | gate x48 | up x48] = 97 -- above the unroll
            # threshold, runs bare (rms resets gate/up counters; every
            # 2nd up block runs one swiglu group).
            for _ in range_(97):
                a = a_f.acquire(1)
                ka(a)
                a_f.release(1)
            # ---- ring 2: gather sw (int8 q + scales) ----
            ob = r2_out.acquire(1)
            kfr2(ob)
            r2_out.release(1)
            for i in range_(48):
                with if_(lt(i, 7), hasElse=False):
                    s = r2_in.acquire(1)
                    kst2(s)
                    with if_(lt(i, 6), hasElse=False):
                        ob = r2_out.acquire(1)
                        kfw2(s, ob)
                        r2_out.release(1)
                    r2_in.release(1)
            # [down x48] (c-major chunk-blocks accumulate f32 into
            # dacc) -- 48 trips, above the unroll threshold, bare.
            for _ in range_(48):
                a = a_f.acquire(1)
                ka(a)
                a_f.release(1)
            # ---- ring 3: gather xn1 = bf16(x' + dacc) in place ----
            # MUST complete before w1: K=102's rms reads the gathered
            # xn1 out of lv_shared (the earlier 49-loop form ran w1
            # before this gather -- rms over stale gate/up slots).
            ob = r13_out.acquire(1)
            kfr3(ob)
            r13_out.release(1)
            for i in range_(48):
                with if_(lt(i, 7), hasElse=False):
                    s = r13_in.acquire(1)
                    kst3(s)
                    with if_(lt(i, 6), hasElse=False):
                        ob = r13_out.acquire(1)
                        kfw1(s, ob)  # same 512B element size as ring1
                        r13_out.release(1)
                    r13_in.release(1)
            # [w1(K=102) | qkv x24 | cxn x16] = 41 live of 48, one
            # guarded loop, stream order preserved: w1's rms consumes
            # the gathered xn1, sets phase = QKV and rebuilds the
            # arena; qkv blocks then write the REAL C elements; cxn
            # drains the own xn1 chunk as C (next exec's residual
            # source). Sections are sibling guards -- an unsigned
            # range check 0 <= i-base < n (a signed lt fires on the
            # negative wrap) instead of nested if/else, which scf's
            # else_ helper makes fragile.
            def rng(i, base, n):
                return cmpi(
                    "ult",
                    subi(i, constant(base, index=True)),
                    constant(n, index=True),
                )

            for i in range_(48):
                with if_(lt(i, N_QKV + N_CXN + 1), hasElse=False):
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
        for w in range(cols)
    ]

    X_taps = [
        TensorAccessPattern(
            tensor_dims=(1, cols * ELEM),
            offset=w * ELEM,
            sizes=[1, 1, 1, ELEM],
            strides=[0, 0, 0, 1],
        )
        for w in range(cols)
    ]
    XN_taps = X_taps  # identical geometry, different tensor
    W_taps = [
        TensorAccessPattern(
            tensor_dims=(1, cols * N_WELEM * ELEM),
            offset=w * N_WELEM * ELEM,
            sizes=[1, 1, 1, N_WELEM * ELEM],
            strides=[0, 0, 0, 1],
        )
        for w in range(cols)
    ]
    # C drain per worker: 40 elements linear (24 qkv then 16 xn1). The
    # host splits the two sections by ring position.
    C_taps = [
        TensorAccessPattern(
            tensor_dims=(1, cols * (N_QKV + N_CXN) * M_INPUT),
            offset=w * (N_QKV + N_CXN) * M_INPUT,
            sizes=[1, 1, N_QKV + N_CXN, M_INPUT],
            strides=[0, 0, M_INPUT, 1],
        )
        for w in range(cols)
    ]

    rt = Runtime()
    with rt.sequence(L3_W_ty, L3_X_ty, L3_XN_ty, L3_C_ty) as (W, X, XN, C):
        rt.start(*workers)
        tg = rt.task_group()
        for w in range(cols):
            rt.fill(A_fifos[w].prod(), X, X_taps[w], task_group=tg)
            rt.fill(A_fifos[w].prod(), XN, XN_taps[w], task_group=tg)
            rt.fill(A_fifos[w].prod(), W, W_taps[w], task_group=tg)
        for w in range(cols):
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
    argparser.add_argument("--cols", type=int, default=8)
    argparser.add_argument("--output-file-path", "-o", type=str, required=True)
    args = argparser.parse_args()
    module = my_layerv2(args.dev, args.cols)
    with open(args.output_file_path, "w") as f:
        f.write(str(module))
