#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""P13 probe: v5 machinery, compute removed.

The full v5 test lands ~30% above the P12 1ch probe on every shape
(e.g. lm_head 3658 vs 2620 us) — either the real mmul compute is not
hidden under the fill stream, or something in the v5 fill machinery
(X-fill insertion, 2D C taps, 18560-B BDs) is slower than the probe's.

This probe runs the UNCHANGED v5 op but flips every WEIGHT block's K
header to garbage: the kernel's guard takes the zero-rows path (~few
cycles, no mmul, no x copy) while the X element stays live (K=0 ->
staging + one zero C element — the real v5 per-op behavior). Identical
wire bytes, identical fills/drains, identical kernel image.

  full v5  - this probe  = per-block compute EXPOSURE (should hide!)
  probe    - P12 1ch     = v5 fill machinery cost vs the probe's
"""

import pytest
import numpy as np
import torch

from iron.operators.w4gemvu.op import AIEW4GEMVU
from iron.operators.w4gemvu import reference
from iron.common.test_utils import run_test

GARBAGE_K = 0x0000BEEF  # != 0 (activation) and != 2048 (weight block)


params = [
    (2048, 2048),
    (12288, 2048),
    (2048, 6144),
    (121088, 2048),
]
names = [f"w4gemvus_{M}x{K}" for M, K in params]
all_params = [pytest.param(*p, id=n) for p, n in zip(params, names)]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
)
@pytest.mark.parametrize("M,K", all_params)
def test_w4gemvu_v5s(M, K, aie_context):
    golden_ref = reference.generate_golden_reference(M=M, K=K)

    # Flip every weight block's K header to garbage (keep the X element
    # in the "vector" buffer live — it is built separately).
    packed = golden_ref["packed_weights"].copy()
    blocks_total = packed.size // reference.ELEM
    for b in range(blocks_total):
        off = b * reference.ELEM + reference.ELEM - 8
        packed[off : off + 4] = np.frombuffer(
            np.uint32(GARBAGE_K).tobytes(), dtype=np.uint8
        )

    operator = AIEW4GEMVU(
        M=M, K=K, num_aie_columns=8, group_size=32, context=aie_context
    )

    input_buffers = {
        "packed_weights": torch.from_numpy(packed),
        "vector": operator.replicate_vector(golden_ref["x"]),
    }
    # Zero-rows path leaves C at whatever the run initializes; zeros in,
    # zeros out (PERF ONLY).
    output_buffers = {
        "output": torch.zeros(
            reference.output_rows(M, K), dtype=golden_ref["output_raw"].dtype
        )
    }

    _, latency_us, _ = run_test(operator, input_buffers, output_buffers)
    stream_mb = len(packed) / 1e6
    print(
        f"\n[v5s {M}x{K}] Latency (us): {latency_us:.1f}, "
        f"{stream_mb / latency_us * 1e3:.2f} GB/s weights"
    )
