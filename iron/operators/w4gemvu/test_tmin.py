#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""P15b probe: machinery floor T_min of the v5 op.

The E2E trace decomposition (P14/P15) says every op pays ~90-100us of
FIXED cost on top of ~0.335us/block (55 GB/s stream). v5s (zero-compute
probe) matched the full op at every shape, so the fixed part is pure
fill machinery, not compute exposure. Candidates: DPU-side BD issue
serialization (24 BD ops per op: 8 X fills + 8 weight fills + 8 C
drains), DMA channel ramp, or per-exec firmware overhead.

This probe shrinks the op to 1-4 blocks/column (M=256..1024, K=2048):
stream time < 3 us/op. Whatever latency remains IS the machinery floor:

  T(M=256)                 = T_min  (exec + issue + ramp + drain wait)
  T(M=2048) - T(M=256)     = 15 blocks/col marginal * ... (stream)
  v5s(M=256)               = same machinery, zero compute (cross-check)

If T_min ~ 90us, the 24-BD-issue serialization model holds (tau ~ 3.5-
4us/BD) and merging the X fill into the weight BD (24 -> 16 BDs) is
worth ~25-30us/op; if T_min is much lower, the fixed cost rides on the
LARGE weight BDs themselves (per-BD ramp scaling with transfer size or
channel arbitration) and the lever is elsewhere.
"""

import pytest
import torch

from iron.operators.w4gemvu.op import AIEW4GEMVU
from iron.operators.w4gemvu.reference import generate_golden_reference
from iron.common.test_utils import run_test


params = [
    (256, 2048),  # 1 block/column — pure machinery
    (512, 2048),  # 2 blocks/col
    (1024, 2048),  # 4 blocks/col
    (2048, 2048),  # 16 blocks/col (o_proj — the E2E anchor)
]
names = [f"w4gemvut_{M}x{K}" for M, K in params]
all_params = [pytest.param(*p, id=n) for p, n in zip(params, names)]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
)
@pytest.mark.parametrize("M,K", all_params)
def test_w4gemvu_tmin(M, K, aie_context):
    golden_ref = generate_golden_reference(M=M, K=K)

    operator = AIEW4GEMVU(
        M=M,
        K=K,
        num_aie_columns=8,
        group_size=32,
        context=aie_context,
    )

    input_buffers = {
        "packed_weights": torch.from_numpy(golden_ref["packed_weights"]),
        "vector": operator.replicate_vector(golden_ref["x"]),
    }
    output_buffers = {
        "output": golden_ref["output_raw"]
    }

    errors, latency_us, _ = run_test(
        operator,
        input_buffers,
        output_buffers,
        rel_tol=0.07,
        abs_tol=0.7,
    )

    blocks_per_col = (M // 8 // 16) * (K // 2048)
    print(
        f"\n[tmin {M}x{K}] blocks/col={blocks_per_col} "
        f"Latency (us): {latency_us:.1f}"
    )

    assert not errors, f"Test failed with errors: {errors}"
