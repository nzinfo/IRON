#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from iron.operators.w4gemv2.op import AIEW4GEMV2
from iron.operators.w4gemv2.reference import generate_golden_reference
from iron.common.test_utils import run_test


# (M, K, cols, tsi, tso, group_size)
params = [
    (2048, 2048, 4, 1, 512, 32),
    (2048, 2048, 4, 16, 512, 32),
    (2048, 2048, 8, 16, 256, 32),
]

names = [f"w4gemv2_{M}x{K}_{tsi}tsi_{tso}tso_{cols}col_g{gs}" for M, K, cols, tsi, tso, gs in params]

all_params = [pytest.param(*p, id=n) for p, n in zip(params, names)]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
)
@pytest.mark.parametrize("M,K,cols,tsi,tso,gs", all_params)
def test_w4gemv2(M, K, cols, tsi, tso, gs, aie_context):
    golden_ref = generate_golden_reference(M=M, K=K, group_size=gs, m_input=tsi, cols=cols)

    operator = AIEW4GEMV2(
        M=M,
        K=K,
        num_aie_columns=cols,
        tile_size_input=tsi,
        tile_size_output=tso,
        group_size=gs,
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

    weight_kb = (M * K // 2 + M * (K // gs) * 2) / 1024
    print(
        f"\n[{tsi}tsi {cols}col] Latency (us): {latency_us:.1f}, "
        f"{weight_kb * 1e3 / latency_us / 1e3:.2f} GB/s weights"
    )

    assert not errors, f"Test failed with errors: {errors}"
