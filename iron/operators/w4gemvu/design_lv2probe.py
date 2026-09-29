# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.

"""P28-6 bisect probe: layerv2 with a LADDER of ring gathers (rings=0..3).

Facts so far (all N=16): the Hamiltonian design compiles; the bare ring
probe passes (test_ring.py); the no-ring flavor chain + A/C plumbing
passes (this probe, rings=0); the FULL design deadlocks. The remaining
delta is the gather relay itself, one ring family at a time:

    rings=1: [X|xn|o] + RING1 (x') + rest no-C + cxn
    rings=2: + RING2 (sw) after gate/up
    rings=3: + RING3 (xn1) after down  (all three rings, count-exact)

The first rung that hangs names the guilty ring. Values up to the last
included ring are real; everything downstream runs on stale slots
(finite, deterministic zeros on iter 1). The C count is always exactly
what the drain expects. The P28-5 artifacts stay untouched.
"""

import numpy as np
from ml_dtypes import bfloat16
import argparse

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_, if_
from aie.extras.dialects.arith import cmpi, constant
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2

from iron.operators.w4gemvu.design_layerv2 import ring_tables

ELEM = 18560
M_INPUT = 16
HIDDEN = 2048
INTER = 6144
QKV_M = 3072


def my_lv2probe(dev, cols=8, rings=0):
    n = cols
    assert n in (8, 16)
    # rings>=10: dumb-kernel discriminator (rings-10 selects the swap:
    # 1=all three, 2=make/fr1, 3=store/st1, 4=forward/fw1, 5=forward
    # SLOW-dumb (spin-padded copy: lv_fw1's timing, not its codegen);
    # the r13 gather's fifo ops stay identical).
    dumb = rings >= 10
    dswap = rings - 10 if dumb else 0
    rings = 1 if dumb else rings
    if dumb:
        assert 1 <= dswap <= 7 and cols == 16, "dumb mode: (16, 1X) only"
    assert 0 <= rings <= 3
    rows = HIDDEN // n
    N_O = rows // 16
    N_GATE = (INTER // n) // 16
    N_UP = N_GATE
    N_DOWN = 3 * N_O
    N_QKV = (QKV_M // n) // 16
    N_CXN = N_O
    N_WELEM = N_O + N_GATE + N_UP + N_DOWN + N_QKV + 2
    grp = (INTER // n) // 32
    R13 = rows
    R2 = (INTER // n) + ((grp * 2 + 31) // 32) * 32
    ROUNDS = n - 1
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
    k_a = Kernel("w4gemvu_layer_a", bin_name, [L1_A_ty])
    k_cxn = Kernel("lv_cxn", bin_name, [L1_C_ty])
    # rings>=10 reaches here decoded (see top of my_lv2probe): the r13
    # gather bodies become ring_probe.o dumb kernels; Pass => lv_* kernel
    # N=16 data path guilty, hang => fill dispatch guilty (perf-lab 6f-2).
    if dumb:
        # one binary per worker (iron constraint); Kernel objects are
        # reusable symbols -- k_t covers both 1-arg call sites.
        k_t = Kernel("ring_touch1", bin_name, [L1_R13_ty])
        k_c = Kernel("ring_copy128_bf16", bin_name, [L1_R13_ty, L1_R13_ty])
        k_fr1 = k_t if dswap in (1, 2) else Kernel("lv_fr1", bin_name, [L1_R13_ty])
        k_st1 = k_t if dswap in (1, 3) else Kernel("lv_st1", bin_name, [L1_R13_ty])
        k_fw1 = (k_c if dswap in (1, 4) else
                 Kernel("ring_copy128_slow", bin_name, [L1_R13_ty, L1_R13_ty])
                 if dswap == 5 else
                 Kernel("lv_fw1_clone", bin_name, [L1_R13_ty, L1_R13_ty])
                 if dswap == 6 else
                 Kernel("ring_fw_const", bin_name, [L1_R13_ty, L1_R13_ty])
                 if dswap == 7 else
                 Kernel("lv_fw1", bin_name, [L1_R13_ty, L1_R13_ty]))
    else:
        k_fr1 = Kernel("lv_fr1", bin_name, [L1_R13_ty])
        k_fw1 = Kernel("lv_fw1", bin_name, [L1_R13_ty, L1_R13_ty])
        k_st1 = Kernel("lv_st1", bin_name, [L1_R13_ty])
    k_fr2 = Kernel("lv_fr2", bin_name, [L1_R2_ty])
    k_fw2 = Kernel("lv_fw2", bin_name, [L1_R2_ty, L1_R2_ty])
    k_st2 = Kernel("lv_st2", bin_name, [L1_R2_ty])
    k_fr3 = Kernel("lv_fr3", bin_name, [L1_R13_ty])
    k_st3 = Kernel("lv_st3", bin_name, [L1_R13_ty])

    A_fifos = [
        ObjectFifo(L1_A_ty, name=f"A_L3L1_{i}", depth=2) for i in range(n)
    ]
    C_fifos = [
        ObjectFifo(L1_C_ty, name=f"C_L1L3_{i}", depth=2) for i in range(n)
    ]
    ring13_fifos = [
        ObjectFifo(L1_R13_ty, name=f"ring13_{w}", depth=2) for w in range(n)
    ]
    ring2_fifos = [
        ObjectFifo(L1_R2_ty, name=f"ring2_{w}", depth=2) for w in range(n)
    ]

    def core_body(a_f, r13_out, r13_in, r2_out, r2_in, c_f,
                  ka, kcxn, kfr1, kfw1, kst1, kfr2, kfw2, kst2, kfr3, kst3):
        def lt(i, n_):
            return cmpi("slt", i, constant(n_, index=True))

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
            consume(2 + N_O)
            if rings >= 1:
                gather(r13_out, r13_in, kfr1, kfw1, kst1)
            consume(1 + N_GATE + N_UP)
            if rings >= 2:
                gather(r2_out, r2_in, kfr2, kfw2, kst2)
            consume(N_DOWN)
            if rings >= 3:
                gather(r13_out, r13_in, kfr3, kfw1, kst3)
            consume(1 + N_QKV)
            for i in range_(48):
                with if_(lt(i, N_QKV + N_CXN), hasElse=False):
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
                k_a,
                k_cxn,
                k_fr1,
                k_fw1,
                k_st1,
                k_fr2,
                k_fw2,
                k_st2,
                k_fr3,
                k_st3,
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
    XN_taps = X_taps
    W_taps = [
        TensorAccessPattern(
            tensor_dims=(1, n * N_WELEM * ELEM),
            offset=w * N_WELEM * ELEM,
            sizes=[1, 1, 1, N_WELEM * ELEM],
            strides=[0, 0, 0, 1],
        )
        for w in range(n)
    ]
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
        # P28-6 FILL-ORDER LAW: all small unblocking fills (X, XN) for
        # EVERY worker first, then the huge W fills. At N=16 two workers
        # share a shim; an interleaved order issues w's XN right after
        # the roommate's 1.74 MB W fill, and the roommate parks at its
        # first gather after consuming only 18/94 W elements -- the W
        # fill never completes, the XN behind it never issues, w never
        # reaches its make-push, and the ring deadlocks cross-worker.
        # N=8 is immune (1 worker/shim: nobody competes for your XN);
        # rings=0 is immune (no gather: everyone drains their own W).
        for w in range(n):
            rt.fill(A_fifos[w].prod(), X, X_taps[w], task_group=tg)
            rt.fill(A_fifos[w].prod(), XN, XN_taps[w], task_group=tg)
        for w in range(n):
            rt.fill(A_fifos[w].prod(), W, W_taps[w], task_group=tg)
        for w in range(n):
            rt.drain(C_fifos[w].cons(), C, C_taps[w], task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(prog="P28-6 lv2 bisect probe")
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("--cols", type=int, default=16)
    argparser.add_argument("--rings", type=int, default=0)
    argparser.add_argument("--output-file-path", "-o", type=str, required=True)
    args = argparser.parse_args()
    module = my_lv2probe(args.dev, args.cols, args.rings)
    with open(args.output_file_path, "w") as f:
        f.write(str(module))
