#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from iron.operators.flowkv_decode.op import AIEFlowKVDecode, pack_q_with_angles
from iron.operators.flowkv_decode.reference import generate_golden_reference
from iron.common.test_utils import run_test


def generate_test_params(extensive=False):
    params = [
        # (num_heads, num_kv_heads, head_dim, seq_len, chunk_size, num_cols)
        (32, 8, 64, 128, 32, 4),
    ]
    if extensive:
        params += [
            (32, 8, 64, 256, 32, 4),
            (32, 8, 64, 512, 32, 8),
            (32, 8, 64, 1024, 32, 4),
            # MiniCPM5-2B decode shape: 16 Q heads, 2 KV heads, d=128.
            # num_kv_heads=2 limits num_cols to {1, 2}.
            (16, 2, 128, 256, 32, 2),
            (16, 2, 128, 1024, 32, 2),
            # hy-mt2 1.8B decode shape: 16 Q heads, 4 KV heads (GQA group 4).
            # 4 cols = 4 shim DMA tiles, inside the npu2 budget.
            (16, 4, 128, 1024, 32, 4),
        ]
    names = [
        f"flowkv_decode_{nh}h_{nkv}kv_{d}d_{s}s_{cs}cs_{nc}col"
        for nh, nkv, d, s, cs, nc in params
    ]
    return params, names


# 8 columns need 8 distinct shim DMA tiles; npu2 (Strix Halo) does not have
# that many usable shim columns, so the SequentialPlacer fails to resolve
# this shape regardless of kernel code. Verified failing at upstream HEAD.
NO_PLACE = {"flowkv_decode_32h_8kv_64d_512s_32cs_8col"}

regular_params, regular_names = generate_test_params(extensive=False)
extensive_params, extensive_names = generate_test_params(extensive=True)

all_params = [
    pytest.param(*params, id=name)
    for params, name in zip(regular_params, regular_names)
] + [
    pytest.param(
        *params,
        marks=(
            [pytest.mark.extensive, pytest.mark.skip(reason="exceeds npu2 shim column budget")]
            if name in NO_PLACE
            else [pytest.mark.extensive]
        ),
        id=name,
    )
    for params, name in zip(extensive_params, extensive_names)
]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim,seq_len,chunk_size,num_cols",
    all_params,
)
def test_flowkv_decode(
    num_heads,
    num_kv_heads,
    head_dim,
    seq_len,
    chunk_size,
    num_cols,
    aie_context,
):
    golden_ref = generate_golden_reference(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
    )

    operator = AIEFlowKVDecode(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        chunk_size=chunk_size,
        num_cols=num_cols,
        context=aie_context,
    )

    group_size = num_heads // num_kv_heads
    q_packed = pack_q_with_angles(
        golden_ref["Q"],
        golden_ref["q_angles"],
        group_size,
        num_kv_heads,
    )

    input_buffers = {
        "kv_cache": golden_ref["KV_interleaved"],
        "queries": q_packed,
    }
    output_buffers = {"output": golden_ref["O"]}

    # Online softmax + bf16 GEMV accumulates rounding error across chunks.
    # abs_tol was 1.0 (the composed-operator ladder), which masked a real
    # bug: a bf16-recursed softmax denominator left outputs scaled
    # ~1.5-1.9x per head on flat-softmax data. With that fixed, the
    # remaining floor is quantization inside the kernel's bf16 score path:
    # scores AND the exp2 argument round to bf16, so at this data's
    # max|s|~14 the argument ULP (~0.08 at diff*log2e ~ -20) perturbs
    # weights by ~5% (the f32-score reference has none of this). Measured
    # worst case 0.17 at one head's four elements; 0.25 floors it with
    # headroom while any whole-output scale drift >= ~25% fails. The
    # flat-softmax test below is the dedicated denominator tripwire at a
    # 6x tighter floor (small scores -> microscopic quantization).
    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        input_buffers,
        output_buffers,
        rel_tol=0.07,
        abs_tol=0.25,
    )

    print(f"\nLatency (us): {latency_us:.1f}")

    # Compute throughput: 2 * num_heads * seq_len * head_dim FLOPs (Q@K + attn@V)
    flops = 2.0 * 2 * num_heads * seq_len * head_dim
    gflops = flops / (latency_us * 1e-6) / 1e9
    print(f"Throughput: {gflops:.6e} GFLOP/s")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"


def test_flowkv_decode_runtime_seq(aie_context):
    """One compiled binary (cache capacity 128) serving several live lengths.

    The runtime sequence length S rides in the Q element header; cache rows
    at positions >= S still stream (the persistent workers' acquire counts
    are compiled in) and must be neutralized by the kernel's dead-row
    sentinel. S=1 is the first decode step (chunk 0 row 0 live, everything
    else dead), S=101 splits the last live chunk mid-way (intra-chunk
    masking), S=128 is the full-capacity regression. Reuses ONE operator
    instance across all S values — the daemon usage pattern.
    """
    import torch

    num_heads, num_kv_heads, head_dim = 16, 2, 128
    cache_seq, chunk_size, num_cols = 128, 32, 2
    group_size = num_heads // num_kv_heads

    operator = AIEFlowKVDecode(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len=cache_seq,
        chunk_size=chunk_size,
        num_cols=num_cols,
        context=aie_context,
    )

    for S in (1, 33, 101, 128):
        golden_ref = generate_golden_reference(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            seq_len=S,
        )
        # Zero-pad the interleaved cache to the compiled capacity: dead V
        # rows must be finite (0 x F_c=0), dead K rows may hold anything.
        live = golden_ref["KV_interleaved"].view(num_kv_heads, S, 2, head_dim)
        padded = torch.zeros(
            num_kv_heads, cache_seq, 2, head_dim,
            dtype=golden_ref["KV_interleaved"].dtype,
        )
        padded[:, :S] = live

        q_packed = pack_q_with_angles(
            golden_ref["Q"],
            golden_ref["q_angles"],
            group_size,
            num_kv_heads,
            seq_len_cur=S,
        )
        errors, _, _ = run_test(
            operator,
            {"kv_cache": padded.reshape(-1), "queries": q_packed},
            {"output": golden_ref["O"]},
            rel_tol=0.07,
            abs_tol=0.25,
        )
        assert not errors, f"S={S} failed with errors: {errors}"


def test_flowkv_decode_flat_softmax(aie_context):
    """Large-denominator regime: K scaled to keep scores ~N(0, 0.6^2) so the
    softmax over a 1024-slot cache stays flat (l ~ O(1000)) while weights
    still vary a few x (O magnitudes ~0.3). The bf16 l-recursion bug lived
    exactly here — each typical f (~0.05) sat below half a bf16 ULP once
    l > ~128 and was silently dropped, shrinking the denominator to ~60%
    and scaling O by 1/l_used (errors ~0.2 here, far past the 0.04 floor).
    The default sharp data (peaked softmax, l ~ O(1)) never exercised it.
    Small scores also keep the bf16 score-ULP noise microscopic, so this
    case runs at the tight 0.04 floor. Same compiled shape as the MiniCPM5
    decode fixture.
    """
    num_heads, num_kv_heads, head_dim = 16, 2, 128
    seq_len, chunk_size, num_cols = 1024, 32, 2
    group_size = num_heads // num_kv_heads

    golden_ref = generate_golden_reference(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        k_val_range=0.6,
    )

    operator = AIEFlowKVDecode(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        chunk_size=chunk_size,
        num_cols=num_cols,
        context=aie_context,
    )

    q_packed = pack_q_with_angles(
        golden_ref["Q"],
        golden_ref["q_angles"],
        group_size,
        num_kv_heads,
    )

    errors, _, _ = run_test(
        operator,
        {"kv_cache": golden_ref["KV_interleaved"], "queries": q_packed},
        {"output": golden_ref["O"]},
        rel_tol=0.07,
        abs_tol=0.04,
    )
    assert not errors, f"flat-softmax case failed with errors: {errors}"
