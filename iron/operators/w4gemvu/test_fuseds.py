#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""P27-3b probe: fused rms-pair machinery, compute removed.

The E2E chain runs pair A (o->gateup) at 460us and pair B (down->qkv) at
350us — 33.6 GB/s aggregate over the layer's 27.2 MB, while FLM's layer
exec sustains 47-56 GB/s on the same wall and the P12 1ch no-B probe
pegged the marginal fill rate at 55.6. Before redesigning the pair
choreography, split the gap:

  full pair (test_fused)  -  this probe  =  per-block compute EXPOSURE
  this probe              -  stream floor = fill/drain machinery cost
                                         (task-group boundary, glue
                                         elements, C drains, BD shape)

Same trick as test_v5s: flip every WEIGHT block's K header to garbage —
the kernel guard takes the zero-rows path (a few cycles, no mmul, no
operand rebuild) while the X element (K=0), the rms window (K=1) and
the w elements (K=3) stay live, so the glue flavors still run. Identical
wire bytes, identical fills/drains, identical kernel image. PERF ONLY:
zeros in, zeros out, no golden assert.
"""

import pytest
import numpy as np
import torch

from iron.operators.w4gemvu.op_fused import AIEW4GEMVUFused, RMS_ELEM_ROWS
from iron.operators.w4gemvu import reference
from iron.operators.w4gemvu.reference import ELEM, TILE_ROWS
from iron.operators.w4gemvu.test_fused import generate_fused_reference
from iron.common.test_utils import run_test

GARBAGE_K = 0x0000BEEF  # != 0 (activation) and != 2048 (weight block)


def _decompute(packed):
    """Zero the compute path of every weight block in a packed stream."""
    packed = packed.copy()
    blocks_total = packed.size // ELEM
    for b in range(blocks_total):
        off = b * ELEM + ELEM - 8
        packed[off : off + 4] = np.frombuffer(
            np.uint32(GARBAGE_K).tobytes(), dtype=np.uint8
        )
    return packed


params = [
    (2048, 2048, 12288),  # o_proj -> rms -> gate_up (the E2E pair)
    (2048, 6144, 3072),  # down_proj -> rms -> qkv(next layer)
]
names = [f"w4gemvufs_{M1}x{K1}_{M2}" for M1, K1, M2 in params]
all_params = [pytest.param(*p, id=n) for p, n in zip(params, names)]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
)
@pytest.mark.parametrize("M1,K1,M2", all_params)
def test_w4gemvu_fuseds(M1, K1, M2, aie_context):
    golden = generate_fused_reference(M1, K1, M2)

    operator = AIEW4GEMVUFused(
        M1=M1, K1=K1, M2=M2, num_aie_columns=8, group_size=32, context=aie_context
    )

    # run_test ZEROES every output buffer, then writes input buffers — the
    # C BO's host-written rms window (residual + K=1/blocks1 headers) must
    # ride in input_buffers or the dispatcher mis-stages the window (P16).
    input_buffers = {
        "packed1": torch.from_numpy(_decompute(golden["packed1"])),
        "packed2": operator.build_packed2(
            torch.from_numpy(_decompute(golden["packed2_blocks"])), golden["wgt"]
        ),
        "vector": golden["activation"],
        "output": operator.build_c_init(golden["res"]),
    }
    blocks1 = reference.blocks_per_col(M1, K1)
    blocks2 = reference.blocks_per_col(M2, 2048)
    c_total = RMS_ELEM_ROWS + 8 * (blocks2 + 2) * TILE_ROWS
    output_buffers = {
        "output": torch.zeros(c_total, dtype=torch.bfloat16)
    }

    _, latency_us, _ = run_test(operator, input_buffers, output_buffers)
    mb = (
        len(golden["packed1"]) + len(golden["packed2_blocks"]) + 8 * ELEM
    ) / 1e6
    print(
        f"\n[fuseds {M1}x{K1}->{M2}] Latency (us): {latency_us:.1f}, "
        f"{mb / latency_us * 1e3:.2f} GB/s weights (PERF ONLY)"
    )
