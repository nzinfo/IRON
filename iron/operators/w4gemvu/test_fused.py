#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""M6 fused rms-pair test (P16): op1 -> add+rms+quantize -> op2 in one
exec, golden computed by mirroring the kernel flavor's scalar semantics
(f32 math, RNE bf16 steps, per-group-32 int8 quantize with d =
bf16(amax/127)). The reference reuses the bit-exact gemv reference for
both projections; only the glue is re-derived here.

Perf comparison target: the SPLIT pair costs op1 + gap + op2 on the
E2E chain (o+gateup: 141 + 18.6 + 346 us; down+qkv: 231 + 24.8 + 153).
The fused op should land under (split total - one machinery floor
~100 us - the gap).
"""

import pytest
import numpy as np
import torch

from iron.operators.w4gemvu.op_fused import AIEW4GEMVUFused, RMS_ELEM_ROWS
from iron.operators.w4gemvu import reference
from iron.operators.w4gemvu.reference import ELEM, TILE_ROWS, TILE_K
from iron.common.test_utils import run_test


def _f32(bf16_bits):
    """(...,) uint16 bf16 bits -> float32 array."""
    return (bf16_bits.astype(np.uint32) << 16).view(np.float32)


def _bf16(f32):
    """float32 -> uint16 bf16 bits with RNE (the kernel's carry trick)."""
    u = f32.astype(np.float32).view(np.uint32)
    rounded = u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _kernel_rsqrt(s):
    """The kernel's 3-iteration Newton rsqrt, in np.float32 (same ops):
    x *= (2 - s*x*x) — NOT 2x - s*x^2 (the missing x factor diverges to
    -6.6e6 and garbles the whole golden, second fused run)."""
    v = np.float32(s)
    magic = np.array(0x5F3759DF, dtype=np.uint32)
    x = (magic - (v.view(np.uint32) >> np.uint32(1))).view(np.float32)
    for _ in range(3):
        x = np.float32(x * (np.float32(2.0) - v * x * x))
    return x


def generate_fused_reference(M1, K1, M2, seed=42):
    torch.manual_seed(seed)
    W1 = (torch.rand(M1, K1, dtype=torch.float32) * 2 - 1).numpy()
    W2 = (torch.rand(M2, 2048, dtype=torch.float32) * 2 - 1).numpy()
    x = (torch.rand(K1, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)
    res = (torch.rand(M1, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)
    wgt = (torch.rand(M1, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)

    packed1, W1_deq = reference.quantize_and_pack(W1)
    packed2_blocks, W2_deq = reference.quantize_and_pack(W2)
    q1, d1, x_deq1 = reference.quantize_vector(x)

    # op1 partials (bit path identical to the unfused golden)
    chunks1 = reference.chunks_per_tile(K1)
    partials1 = torch.empty(chunks1, M1, dtype=torch.bfloat16)
    W1f = W1_deq.to(torch.float32)
    for c in range(chunks1):
        k0 = c * TILE_K
        partials1[c] = (W1f[:, k0 : k0 + TILE_K] @ x_deq1[k0 : k0 + TILE_K]).to(
            torch.bfloat16
        )

    # ---- kernel K=3 glue semantics, mirrored ----
    # h2 = bf16(f32(res) + sum_chunks f32(partial))  [no intermediate
    # rounding of the sum — one fewer round than the host glue]
    p_bits = partials1.view(torch.uint16).numpy().astype(np.uint32)  # (chunks, M1)
    res_bits = res.view(torch.uint16).numpy().astype(np.uint32)
    h2f = _f32(res_bits) + _f32(p_bits).sum(axis=0)
    h2_bits = _bf16(h2f)
    h2f_r = _f32(h2_bits)
    sumsq = np.float32(0)
    for v in h2f_r:  # sequential f32 accumulate, kernel/host order
        sumsq = np.float32(sumsq + np.float32(v * v))
    inv = _kernel_rsqrt(np.float32(sumsq / np.float32(M1) + np.float32(1e-5)))

    # xn (f32, never rounded to bf16) -> per-group-32 quantize
    w_bits = wgt.view(torch.uint16).numpy().astype(np.uint32)
    xn = h2f_r * inv * _f32(w_bits)
    xn = xn.reshape(M1 // 32, 32)
    amax = np.abs(xn).max(axis=1)
    d2_bits = _bf16((amax / np.float32(127.0)).astype(np.float32))
    invd = (np.float32(1.0) / _f32(d2_bits.astype(np.uint32)))[:, None]
    v = xn * invd
    v = np.where(amax[:, None] == 0, 0.0, v)  # zero groups: q=0, d=0
    q2 = np.round(v)  # np.round = ties-to-even, the kernel's tie rule
    q2 = np.clip(q2, -127, 127).astype(np.int8).reshape(M1)

    # op2 gemv from the kernel-quantized activation
    d_rep = np.repeat(_f32(d2_bits.astype(np.uint32)), 32)
    x_deq2 = q2.astype(np.float32) * d_rep
    W2f = W2_deq.to(torch.float32)
    partials2 = (W2f @ torch.from_numpy(x_deq2)).to(torch.bfloat16)

    # expected C tensor: [op1 sections | residual | headers | 8 op2
    # sections] — the window's host-written rows survive untouched.
    blocks1 = reference.blocks_per_col(M1, K1)
    blocks2 = reference.blocks_per_col(M2, 2048)
    c_rows1 = 8 * (blocks1 + 2) * TILE_ROWS
    section2 = (blocks2 + 2) * TILE_ROWS
    c_total = RMS_ELEM_ROWS + 8 * section2
    c_exp = torch.zeros(c_total, dtype=torch.bfloat16)
    c_exp[0:c_rows1] = reference.shuffle_output(partials1, M1, K1)
    c_exp[c_rows1 : c_rows1 + M1] = res
    # the rms window's header words (u32 K=1 at ELEM-8, u32 blocks1 at
    # ELEM-4 -> u16 rows 9276..9279 = [1, 0, blocks1, 0])
    bits = c_exp.view(torch.uint16)
    bits[9276] = 1
    bits[9277] = 0
    bits[9278] = blocks1 & 0xFFFF
    bits[9279] = 0
    rows_per_col2 = M2 // 8
    for col in range(8):
        base = RMS_ELEM_ROWS + col * section2 + 2 * TILE_ROWS  # 2 glue dummies
        c_exp[base : base + rows_per_col2] = partials2[
            col * rows_per_col2 : (col + 1) * rows_per_col2
        ]

    return {
        "packed1": packed1,
        "packed2_blocks": packed2_blocks,
        "x": x,
        "res": res,
        "wgt": wgt,
        "activation": reference.build_activation_element(q1, d1, K1),
        "output_raw": c_exp,
        "output": partials2,
        "h2_bits": h2_bits,
        "q2": q2,
    }


params = [
    (2048, 2048, 12288),  # o_proj -> rms -> gate_up (the E2E pair)
    (2048, 6144, 3072),  # down_proj -> rms -> qkv(next layer)
]
names = [f"w4gemvuf_{M1}x{K1}_{M2}" for M1, K1, M2 in params]
all_params = [pytest.param(*p, id=n) for p, n in zip(params, names)]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
)
@pytest.mark.parametrize("M1,K1,M2", all_params)
def test_w4gemvu_fused(M1, K1, M2, aie_context):
    golden = generate_fused_reference(M1, K1, M2)

    operator = AIEW4GEMVUFused(
        M1=M1, K1=K1, M2=M2, num_aie_columns=8, group_size=32, context=aie_context
    )

    # run_test ZEROES every output buffer, then writes input buffers — so
    # the C BO's host-written region (the rms window's residual + K=1/
    # blocks1 header words) must ride in input_buffers or the kernel's
    # dispatcher sees a K=0 header on the window element and stages it as
    # a garbage activation instead of running fused_stage1 (first run's
    # whole-tensor mismatch, P16).
    input_buffers = {
        "packed1": torch.from_numpy(golden["packed1"]),
        "packed2": operator.build_packed2(
            torch.from_numpy(golden["packed2_blocks"]), golden["wgt"]
        ),
        "vector": golden["activation"],
        "output": operator.build_c_init(golden["res"]),
    }
    # The EXPECTED post-run C: [op1 sections | residual | headers | op2
    # sections] — not the pre-run seed.
    output_buffers = {"output": golden["output_raw"]}

    errors, latency_us, _ = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.07, abs_tol=0.7
    )

    mb = (
        len(golden["packed1"])
        + len(golden["packed2_blocks"])
        + 8 * ELEM  # the w elements
    ) / 1e6
    print(
        f"\n[fused {M1}x{K1}->{M2}] Latency (us): {latency_us:.1f}, "
        f"{mb / latency_us * 1e3:.2f} GB/s weights"
    )

    assert not errors, f"Test failed with errors: {errors}"
