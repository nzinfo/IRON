#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch

from iron.operators.rescale.op import AIERescale
from iron.operators.rescale.reference import generate_golden_reference
from iron.common.test_utils import run_test


def generate_test_params():
    #   M,    N, tile_m
    params = [
        (192, 64, 32),
        (256, 128, 32),
    ]
    names = [f"rescale_{M}x{N}_{t}t" for M, N, t in params]
    return params, names


params, names = generate_test_params()

all_params = [pytest.param(*p, id=n) for p, n in zip(params, names)]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize("M,N,tile_m", all_params)
def test_rescale(M, N, tile_m, aie_context):
    golden_ref = generate_golden_reference(M=M, N=N)

    operator = AIERescale(
        M=M,
        N=N,
        tile_m=tile_m,
        context=aie_context,
    )

    input_buffers = {
        "input": golden_ref["input"].reshape(-1),
        # One concatenated f32 scale chunk per row block:
        # [sa rows of this block | full sw column scales].
        "scales": torch.cat(
            [
                torch.cat(
                    [
                        golden_ref["sa"][b * tile_m : (b + 1) * tile_m].to(torch.float32),
                        golden_ref["sw"].to(torch.float32),
                    ]
                )
                for b in range(M // tile_m)
            ]
        ),
    }
    output_buffers = {
        "output": golden_ref["output"].reshape(-1),
    }

    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        input_buffers,
        output_buffers,
        rel_tol=0.005,
        abs_tol=0.005,
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed"
