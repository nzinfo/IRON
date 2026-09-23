#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from iron.operators.w4gemvu.op import AIEW4GEMVU
from iron.operators.w4gemvu.reference import generate_golden_reference
from iron.common.test_utils import run_test


# (M, K): the four MiniCPM5 decode projection shapes plus one small case.
params = [
    (2560, 2048),
    (2048, 2048),
    (12288, 2048),
    (2048, 6144),
]

names = [f"w4gemvu_{M}x{K}" for M, K in params]

all_params = [pytest.param(*p, id=n) for p, n in zip(params, names)]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
)
@pytest.mark.parametrize("M,K", all_params)
def test_w4gemvu(M, K, aie_context):
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
        "vector": golden_ref["x"],
    }
    output_buffers = {"output": golden_ref["output"]}

    errors, latency_us, _ = run_test(
        operator,
        input_buffers,
        output_buffers,
        rel_tol=0.07,
        abs_tol=0.7,
    )

    weight_kb = (M * K // 2 + M * (K // 32) * 2) / 1024
    print(
        f"\n[{K}] Latency (us): {latency_us:.1f}, "
        f"{weight_kb * 1e3 / latency_us / 1e3:.2f} GB/s weights"
    )

    assert not errors, f"Test failed with errors: {errors}"
