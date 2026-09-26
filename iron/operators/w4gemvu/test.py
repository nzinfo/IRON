#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from iron.operators.w4gemvu.op import AIEW4GEMVU
from iron.operators.w4gemvu.reference import (
    generate_golden_reference,
    shuffle_output,
)
from iron.common.test_utils import run_test


# (M, K): the MiniCPM5 + hy-mt2 decode projection shapes plus one small
# case, PADDED to the v3 block ABI (K=2048 -> M % 192 == 0, K=6144 ->
# M % 64 == 0). Same PDI for every shape — only the ctrl code carries
# M and K.
params = [
    (3072, 2048),  # hy-mt2 qkv: cat(q 2048, k 512, v 512), 16Q/4KV GQA
    (2112, 2048),  # o_proj, padded 2048 -> 2112 (tiles pack 3/block)
    (12288, 2048),  # gate_up: cat(gate 6144, up 6144)
    (2048, 6144),  # down_proj (1 tile/block + 2 zero-row calls)
    (2688, 2048),  # small case, padded 2560 -> 2688
    # hy-mt2 lm_head: vocab 120818 padded 120960 (tied embed). Same PDI
    # as every other shape — only the ctrl code carries M.
    (120960, 2048),
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
        # The vector buffer holds F padded copies of x (one per B fifo
        # element) — a bare (K,) tensor would leave slots 2..F stale.
        "vector": operator.replicate_vector(golden_ref["x"]),
    }
    # The device buffer carries every produced C row (K=6144 has 2/3
    # interleaved zeros — real rows first in each 12-row group).
    output_buffers = {
        "output": shuffle_output(golden_ref["output"], M, K)
    }

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
