# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-FileCopyrightText: Copyright (C) 2026 nzinfo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.

"""P28-12 device-side attention vehicle: the layer-v3 numerics core on
the standard 8-worker grid, NO rings — the smallest board-testable form
of "all inference compute on the NPU" (rope + qk-norm + KV streaming +
GQA attention + online softmax in w4gemvu_attn_* flavors).

Per exec per worker the A fifo delivers khist+18 elements:

    [X(K=0) | attn-init(K=210) | kvhist x khist (K=211) | out x16 (K=212)]

and produces 16 C elements (256 bf16 = the worker's two Q heads'
attention output rows). Worker p (ring position) computes Q heads 2p,
2p+1 against KV head p/2 (GQA 16Q/4KV, d=128). The .o is the shared
w4gemvu_layer.o — the micro-dispatcher entries (w4gemvu_attn_a/_bf16)
keep the unused layer flavors out of the 16KB program memory
(gc-sections); runtime S rides the init element (kvhist count is
compile-time here, the kernel's S word must equal khist+1).
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

ELEM = 18560
M_INPUT = 16          # C fifo element = 16 bf16 (32B)
HIDDEN = 2048
HEAD_DIM = 128
HEADS = 16
KV_HEADS = 4


def my_lv2attn(dev, cols=8, khist=47):
    n = cols
    assert n == 8, "the attention vehicle is N=8 (2 heads/worker)"
    assert khist >= 1

    dtype_in = np.dtype[np.uint8]
    dtype_out = np.dtype[bfloat16]
    dev_ty = NPU1() if dev == "npu" else NPU2()

    L1_A_ty = np.ndarray[(ELEM,), dtype_in]
    L1_C_ty = np.ndarray[(M_INPUT,), dtype_out]
    L3_W_ty = np.ndarray[(n * (khist + 18) * ELEM,), dtype_in]
    L3_C_ty = np.ndarray[(n * 16 * M_INPUT,), dtype_out]

    bin_name = "w4gemvu_layer.o"
    k_a = Kernel("w4gemvu_attn_a", bin_name, [L1_A_ty])
    k_bf = Kernel("w4gemvu_attn_bf16", bin_name, [L1_A_ty, L1_C_ty])

    A_fifos = [
        ObjectFifo(L1_A_ty, name=f"A_L3L1_{i}", depth=2) for i in range(n)
    ]
    C_fifos = [
        ObjectFifo(L1_C_ty, name=f"C_L1L3_{i}", depth=2) for i in range(n)
    ]

    def core_body(a_f, c_f, ka, kbf):
        def lt(i, n_):
            return cmpi("slt", i, constant(n_, index=True))

        total = 2 + khist  # X | init | hist x khist
        if total >= 48:
            for _ in range_(total):
                a = a_f.acquire(1)
                ka(a)
                a_f.release(1)
        else:
            for i in range_(48):
                with if_(lt(i, total), hasElse=False):
                    a = a_f.acquire(1)
                    ka(a)
                    a_f.release(1)
        # 16 output elements through the C-producing entry
        for i in range_(48):
            with if_(lt(i, 16), hasElse=False):
                a = a_f.acquire(1)
                c = c_f.acquire(1)
                kbf(a, c)
                c_f.release(1)
                a_f.release(1)

    workers = [
        Worker(
            core_body,
            [A_fifos[w].cons(), C_fifos[w].prod(), k_a, k_bf],
        )
        for w in range(n)
    ]

    W_taps = [
        TensorAccessPattern(
            tensor_dims=(1, n * (khist + 18) * ELEM),
            offset=w * (khist + 18) * ELEM,
            sizes=[1, 1, 1, (khist + 18) * ELEM],
            strides=[0, 0, 0, 1],
        )
        for w in range(n)
    ]
    C_taps = [
        TensorAccessPattern(
            tensor_dims=(1, n * 16 * M_INPUT),
            offset=w * 16 * M_INPUT,
            sizes=[1, 1, 16, M_INPUT],
            strides=[0, 0, M_INPUT, 1],
        )
        for w in range(n)
    ]

    rt = Runtime()
    with rt.sequence(L3_W_ty, L3_C_ty) as (W, C):
        rt.start(*workers)
        tg = rt.task_group()
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
    argparser = argparse.ArgumentParser(prog="P28-12 attention vehicle")
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("--cols", type=int, default=8)
    argparser.add_argument("--khist", type=int, default=47)
    argparser.add_argument("--output-file-path", "-o", type=str, required=True)
    args = argparser.parse_args()
    module = my_lv2attn(args.dev, args.cols, args.khist)
    with open(args.output_file_path, "w") as f:
        f.write(str(module))
