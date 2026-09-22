# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
from pathlib import Path
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib.tap import TensorAccessPattern


def my_rescale(dev, M, N, tile_m, trace_size):
    """q8 route-A epilogue: C[M,N] = bf16(f32(A[M,N]) * sa[M] * sw[N]).

    A is a row-major int32 GEMM accumulation; sa/sw are per-row / per-column
    bf16 scales. One worker consumes row blocks of `tile_m` rows per kernel
    call, so A's 1D access pattern stays contiguous. The scale vectors are
    small and per-block constants, so the host simply tiles them to one copy
    per row block and they flow through plain 1D patterns too.
    """
    if M % tile_m != 0:
        raise ValueError(f"Rows ({M}) must be a multiple of tile_m ({tile_m}).")
    num_blocks = M // tile_m
    if N % 32 != 0:
        raise ValueError(f"Columns ({N}) must be a multiple of the kernel's 32-lane vector.")

    a_ty = np.ndarray[(M * N,), np.dtype[np.int32]]
    at_ty = np.ndarray[(tile_m * N,), np.dtype[np.int32]]
    # Scales flow as one concatenated f32 block per row chunk — [tile_m row
    # scales | N column scales] — keeping the tile at two input DMA channels
    # (a + scales) like the elementwise ops. f32 (not bf16) keeps the kernel
    # on native float vector loads; the host widens the bf16 scales
    # losslessly. The host tiles the chunks back-to-back into one long
    # tensor so a plain 1D pattern slices them.
    block_s = tile_m + N
    s_ty = np.ndarray[(block_s * num_blocks,), np.dtype[np.float32]]
    st_ty = np.ndarray[(block_s,), np.dtype[np.float32]]
    c_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]
    ct_ty = np.ndarray[(tile_m * N,), np.dtype[bfloat16]]

    of_a = ObjectFifo(at_ty, name="a")
    of_s = ObjectFifo(st_ty, name="s")
    of_c = ObjectFifo(ct_ty, name="c")

    rescale_kernel = Kernel(
        "rescale_i32_bf16_vector",
        "rescale.o",
        [at_ty, st_ty, ct_ty, np.int32, np.int32],
    )

    def core_body(of_a, of_s, of_c, rescale):
        for _ in range_(num_blocks):
            elem_a = of_a.acquire(1)
            elem_s = of_s.acquire(1)
            elem_c = of_c.acquire(1)
            rescale(elem_a, elem_s, elem_c, tile_m, N)
            of_a.release(1)
            of_s.release(1)
            of_c.release(1)

    my_worker = Worker(core_body, [of_a.cons(), of_s.cons(), of_c.prod(), rescale_kernel])

    def chunk_pattern(total, chunk):
        # The third transformation dimension iterates over the chunks
        # (sizes/strides collapse over itertools.product in taplib, so a
        # [1,1,1,chunk] pattern would only ever cover the first chunk — the
        # elementwise fixtures never hit this because each worker consumes
        # exactly one chunk there). The NPU shim BD size field is 10 bits,
        # so chunks wider than 1023 elements stack the remainder onto the
        # iteration dimension instead (same access set: total contiguous
        # elements in chunk order; 512 keeps every dimension in range).
        assert chunk % 512 == 0 or chunk < 512, f"chunk {chunk} not 512-friendly"
        if chunk <= 512:
            sizes, strides = [1, 1, total // chunk, chunk], [0, 0, chunk, 1]
        else:
            assert total % 512 == 0
            sizes = [1, 1, total // 512, 512]
            strides = [0, 0, 512, 1]
        return TensorAccessPattern((1, total), 0, sizes, strides)

    rt = Runtime()
    with rt.sequence(a_ty, s_ty, c_ty) as (A, S, C):
        rt.start(my_worker)
        tg = rt.task_group()
        rt.fill(of_a.prod(), A, chunk_pattern(M * N, tile_m * N), task_group=tg)
        rt.fill(of_s.prod(), S, chunk_pattern(block_s * num_blocks, block_s), task_group=tg)
        rt.drain(of_c.cons(), C, chunk_pattern(M * N, tile_m * N), wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":

    def str_to_device(device: str):
        if device == "npu":
            return NPU1()
        elif device == "npu2":
            return NPU2()
        else:
            raise ValueError(f"Device name {device} is unknown.")

    p = argparse.ArgumentParser()
    p.add_argument(
        "-d",
        "--dev",
        required=True,
        dest="device",
        help="AIE Device",
        type=str_to_device,
    )
    p.add_argument("-m", "--rows", required=True, dest="rows", help="Matrix rows M")
    p.add_argument("-n", "--cols", required=True, dest="cols", help="Matrix columns N")
    p.add_argument(
        "-tm",
        "--tile-rows",
        required=False,
        dest="tile_rows",
        default="32",
        help="Rows per kernel block (tile_m)",
    )
    p.add_argument(
        "-t", "--trace-size", required=True, dest="trace_size", help="Trace size"
    )
    p.add_argument(
        "--output-file-path",
        "-o",
        type=str,
        help="Output file path for the generated MLIR module",
    )

    opts = p.parse_args(sys.argv[1:])

    my_rescale(
        opts.device,
        int(opts.rows),
        int(opts.cols),
        int(opts.tile_rows),
        int(opts.trace_size),
    )
