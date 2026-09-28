# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""P28-3 probe: 8 persistent workers in a RING of core<->core ObjectFifos.

Layer-v2 (one task group per layer) needs cross-column redistribution
(rms wants the full 2048-vector, swiglu the full 6144, rms1' the whole
down output) riding core-to-core streams, because the persistent-worker
model has no mid-flow TCT for a DDR round trip. This probe compiles and
runs the minimal topology that fact depends on:

    worker i:  [shim A cons, ring_out prod, ring_in cons, shim C prod]
               = exactly 2 stream-in + 2 stream-out per core.

Topology (serpentine). SequentialPlacer maps workers 0..7 to tiles
(0,2)..(0,5),(1,2)..(1,5) (verified on the swiglu_fused_decode npu_insts
built by this same toolchain). Core<->core ObjectFifo buffers+locks lower
onto the PRODUCER tile with the consumer accessing them remotely, so
every ring edge is chosen nearest-neighbor under that placement:

    0 -> 1 -> 2 -> 3 -> 7 -> 6 -> 5 -> 4 -> 0
    (0,2) (0,3) (0,4) (0,5) | (1,5) (1,4) (1,3) (1,2)
     vertical neighbors          horizontal wrap edges 3->7 and 4->0

Semantics are exec-local deterministic: each exec fills ONE A element
per worker and drains ONE C element per worker. The worker first feeds
the ring (rout = a) and ONLY THEN consumes ring_in -- a producer acquire
waits merely for a free buffer (depth-2 init satisfies it), so the first
lap self-starts with no circular wait. Every exec produces and consumes
exactly one element per ring fifo, so the ring is empty again at the
exec boundary: c_w = a_{pred(w)} from the SAME exec, every exec.

Golden (host side): C[w*chunk:(w+1)*chunk] == A[pred(w)*chunk:...].
A wrong constant names exactly which worker's data arrived.
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

CHUNK = 512  # bf16 elements per ring/fifo element (1 KiB, same as swiglu inter)
SUCC = {0: 1, 1: 2, 2: 3, 3: 7, 7: 6, 6: 5, 5: 4, 4: 0}  # serpentine ring
PRED = {v: k for k, v in SUCC.items()}


def my_ring_probe(dev, cols=8, chunk=CHUNK):
    assert cols == 8, "serpentine tables are for the 8-core npu2 partition"

    dtype = np.dtype[bfloat16]

    dev_ty = NPU1() if dev == "npu" else NPU2()

    L1_ty = np.ndarray[(chunk,), dtype]
    L3_ty = np.ndarray[(cols * chunk,), dtype]

    ring_copy = Kernel("ring_copy_bf16", "ring_probe.o", [L1_ty, L1_ty])

    A_fifos = [
        ObjectFifo(L1_ty, name=f"A_L3L1_{i}", depth=2) for i in range(cols)
    ]
    C_fifos = [
        ObjectFifo(L1_ty, name=f"C_L1L3_{i}", depth=2) for i in range(cols)
    ]
    # ring_fifos[j] carries the edge j -> SUCC[j]
    ring_fifos = [
        ObjectFifo(L1_ty, name=f"ring_{j}", depth=2) for j in range(cols)
    ]

    def core_body(a_fifo, rout_fifo, rin_fifo, c_fifo, fn):
        for _ in range_(0xFFFFFFFF):
            # 1) feed the ring FIRST: producer acquire only waits for a
            #    free buffer, so the first lap self-starts.
            a = a_fifo.acquire(1)
            rout = rout_fifo.acquire(1)
            fn(a, rout)
            rout_fifo.release(1)
            a_fifo.release(1)
            # 2) consume what the predecessor sent THIS exec (the ring was
            #    empty at the exec boundary) and parrot it to the drain.
            rin = rin_fifo.acquire(1)
            c = c_fifo.acquire(1)
            fn(rin, c)
            rin_fifo.release(1)
            c_fifo.release(1)

    workers = [
        Worker(
            core_body,
            [
                A_fifos[i].cons(),
                ring_fifos[i].prod(),
                ring_fifos[PRED[i]].cons(),
                C_fifos[i].prod(),
                ring_copy,
            ],
        )
        for i in range(cols)
    ]

    A_taps = [
        TensorAccessPattern(
            tensor_dims=(1, cols * chunk),
            offset=col * chunk,
            sizes=[1, 1, 1, chunk],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]
    C_taps = A_taps  # identical geometry

    rt = Runtime()
    with rt.sequence(L3_ty, L3_ty) as (A, C):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(cols):
            rt.fill(A_fifos[i].prod(), A, A_taps[i], task_group=tg)
        for i in range(cols):
            rt.drain(C_fifos[i].cons(), C, C_taps[i], task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(prog="P28-3 ring probe")
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("--cols", type=int, default=8)
    argparser.add_argument("--output-file-path", "-o", type=str, required=True)
    args = argparser.parse_args()
    module = my_ring_probe(args.dev, args.cols)
    with open(args.output_file_path, "w") as f:
        f.write(str(module))
