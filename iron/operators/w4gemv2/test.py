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
    # The three MiniCPM5 decode GEMV shapes (8col/16tsi, the engine config):
    # fused qkv (2560 = q 2048 + k 256 + v 256), fused gate+up (2*6144),
    # down projection (K = intermediate 6144).
    (2560, 2048, 8, 16, 320, 32),
    (12288, 2048, 8, 16, 1536, 32),
    # down projection: the per-tile data-memory budget is 64 KB (placement
    # map: A fifo 2x27648 + B 12288 + C 1024 = 68 KB overflows at tsi=8);
    # tsi=4 tiles are 13824 B -> ~42 KB total.
    (2048, 6144, 8, 4, 256, 32),
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
